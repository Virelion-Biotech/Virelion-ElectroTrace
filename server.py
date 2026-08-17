"""ElectroTrace research API server."""
from __future__ import annotations
import os, re, sys, tempfile, uuid, zipfile
from pathlib import Path
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from scipy import signal as sps
ROOT=Path(__file__).resolve().parent; SRC=ROOT/"src"
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
from electrotrace.beats import segment_beats
from electrotrace.benchmark import benchmark_models
from electrotrace.formats import MAX_ARCHIVE_BYTES,MAX_ARCHIVE_MEMBERS,MAX_MEMBER_BYTES,load_electrophysiology
from electrotrace.io import load_csv,validate_dataframe
from electrotrace.metadata import recording_metadata
from electrotrace.ml import rank_uncertain,train_classifier
from electrotrace.phenotype import beat_phenotypes,summary_statistics
from electrotrace.project_store import ProjectStore,RecordingRef
from electrotrace.signal import FilterConfigurationError,apply_pipeline
from electrotrace.statistics import benjamini_hochberg,compare_groups
from electrotrace.window import read_recording_window
WEB=ROOT/"web"; SAMPLE_DATA=ROOT/"sample_data"
UPLOAD_ROOT=Path(os.getenv("ELECTROTRACE_UPLOAD_ROOT",Path(tempfile.gettempdir())/"electrotrace_uploads")); PROJECT_ROOT=Path(os.getenv("ELECTROTRACE_PROJECT_ROOT",ROOT/"projects")).resolve()
UPLOAD_ROOT.mkdir(parents=True,exist_ok=True); PROJECT_ROOT.mkdir(parents=True,exist_ok=True)
app=Flask(__name__,static_folder=str(WEB),static_url_path=""); app.config["MAX_CONTENT_LENGTH"]=MAX_ARCHIVE_BYTES
MAX_JSON_BODY_BYTES=64*1024*1024

def _json():
    length=request.content_length
    if length is not None and length>MAX_JSON_BODY_BYTES:
        raise ValueError("JSON request exceeds the 64 MB limit; use recording window endpoints for large signals")
    return request.get_json(silent=True) or {}
def _project_store(name):
    name=str(name).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}",name): raise ValueError("invalid project name")
    return ProjectStore((PROJECT_ROOT/name).resolve())

def _safe_extract(archive,root):
    members=archive.infolist(); root=Path(root).resolve(); total=0
    if len(members)>MAX_ARCHIVE_MEMBERS: raise ValueError("WFDB ZIP contains too many files")
    for m in members:
        if not m.filename or m.filename.endswith('/'): continue
        if m.file_size<0 or m.file_size>MAX_MEMBER_BYTES: raise ValueError("WFDB ZIP contains an oversized member")
        total+=m.file_size
        if total>MAX_ARCHIVE_BYTES: raise ValueError("WFDB ZIP uncompressed size exceeds the limit")
        target=(root/m.filename).resolve()
        if target!=root and root not in target.parents: raise ValueError("Unsafe archive path")
        if ((m.external_attr>>16)&0o170000)==0o120000: raise ValueError("WFDB ZIP symlinks are not supported")
    archive.extractall(root)

def _save_upload(f):
    token=uuid.uuid4().hex; suffix=Path(f.filename or "recording.bin").suffix.lower(); dst=UPLOAD_ROOT/f"{token}{suffix}"; f.save(dst)
    if dst.stat().st_size>MAX_ARCHIVE_BYTES: dst.unlink(missing_ok=True); raise ValueError("uploaded file exceeds the limit")
    if suffix not in {'.zip','.wfdb'}: return token,dst
    d=UPLOAD_ROOT/token; d.mkdir()
    try:
        with zipfile.ZipFile(dst) as z:
            if z.testzip() is not None: raise ValueError("WFDB ZIP is corrupted")
            _safe_extract(z,d)
        hs=list(d.rglob('*.hea'))
        if len(hs)!=1: raise ValueError("WFDB ZIP must contain exactly one .hea header")
        dst.unlink(missing_ok=True); return token,hs[0].with_suffix('')
    except Exception:
        for p in sorted(d.rglob('*'),reverse=True):
            if p.is_file() or p.is_symlink(): p.unlink(missing_ok=True)
            elif p.is_dir(): p.rmdir()
        d.rmdir(); dst.unlink(missing_ok=True); raise

@app.errorhandler(413)
def too_large(_): return jsonify({"error":"request exceeds the 512 MB limit"}),413
@app.errorhandler(ValueError)
def bad_value(e): return jsonify({"error":str(e)}),400
@app.get('/')
def index(): return send_from_directory(str(WEB),'index.html')
@app.get('/sample_data/<path:name>')
def sample_data(name): return send_from_directory(str(SAMPLE_DATA),name)

@app.post('/api/analyze')
def analyze():
    f=request.files.get('file')
    if f is None:return jsonify({"error":"No recording uploaded."}),400
    fn=f.filename or 'recording.csv'
    try:
        raw=f.read()
        if len(raw)>64*1024*1024: raise ValueError('analyze endpoint is limited to 64 MB; use /api/recording and window access for larger recordings')
        if Path(fn).suffix.lower()=='.csv':
            df=load_csv(raw); r=validate_dataframe(df,request.form.get('time_col') or None)
            out={"valid":r.valid,"errors":r.errors,"warnings":r.warnings,"time_col":r.time_col,"signal_cols":r.signal_cols,"sampling_rate_hz":r.sampling_rate_hz,"duration_s":r.duration_s,"n_samples":r.n_samples,"time_start_s":float(df[r.time_col].iloc[0]) if r.valid else None,"time_end_s":float(df[r.time_col].iloc[-1]) if r.valid else None,"format":"csv","source_format":"CSV","filename":fn}
            if r.valid: out["time"]=df[r.time_col].astype(float).tolist(); out["signals"]={c:df[c].astype(float).to_numpy().tolist() for c in r.signal_cols}
            return jsonify(out)
        out=load_electrophysiology(raw,fn); out.update(valid=True,errors=[],warnings=[],format=out['source_format'].lower(),filename=fn,time_start_s=out['time'][0],time_end_s=out['time'][-1]); return jsonify(out)
    except Exception as e:return jsonify({"error":f"Could not read recording: {e}"}),400

@app.post('/api/recording')
def recording_upload():
    f=request.files.get('file')
    if f is None:return jsonify({"error":"No recording uploaded."}),400
    token=path=None
    try:
        token,path=_save_upload(f); meta=recording_metadata(path)
        return jsonify({"recording_id":token,"filename":f.filename,"format":meta["source_format"],"sampling_rate_hz":meta["sampling_rate_hz"],"duration_s":meta["duration_s"],"time_start_s":meta["time_start_s"],"time_end_s":meta["time_end_s"],"n_samples":meta["n_samples"],"channels":meta["channels"]})
    except Exception as e:
        if path and path.is_file(): path.unlink(missing_ok=True)
        if token and (UPLOAD_ROOT/token).is_dir():
            for p in sorted((UPLOAD_ROOT/token).rglob('*'),reverse=True):
                if p.is_file() or p.is_symlink(): p.unlink(missing_ok=True)
                elif p.is_dir(): p.rmdir()
            (UPLOAD_ROOT/token).rmdir()
        return jsonify({"error":str(e)}),400

@app.get('/api/recording/<recording_id>/window')
def recording_window(recording_id):
    if not re.fullmatch(r'[0-9a-f]{32}',recording_id): return jsonify({"error":"invalid recording id"}),400
    try:
        candidates=[UPLOAD_ROOT/recording_id,*UPLOAD_ROOT.glob(recording_id+'.*')]; path=next((p for p in candidates if p.exists()),None)
        if path is None and (UPLOAD_ROOT/recording_id).is_dir():
            hs=list((UPLOAD_ROOT/recording_id).rglob('*.hea')); path=hs[0].with_suffix('') if len(hs)==1 else None
        if path is None:return jsonify({"error":"recording not found"}),404
        start=int(request.args.get('start',0)); stop=int(request.args.get('stop',start+5000)); w=read_recording_window(path,start,stop)
        return jsonify({"start":w['start'],"stop":w['stop'],"n_samples":w['n_samples'],"total_samples":w.get('total_samples'),"sampling_rate_hz":w['sampling_rate_hz'],"time_start_s":w['time_start_s'],"time_end_s":w['time_end_s'],"time":w['time'].tolist(),"signals":{k:np.asarray(v).tolist() for k,v in w['signals'].items()}})
    except (TypeError,ValueError,RuntimeError) as e:return jsonify({"error":str(e)}),400

@app.post('/api/filter')
def filter_signal():
    d=_json()
    try:
        y=apply_pipeline(np.asarray(d['signal'],float),float(d['sampling_rate_hz']),baseline=bool(d.get('baseline')),highpass_hz=float(d['highpass_hz']) if d.get('highpass_hz') is not None else None,lowpass_hz=float(d['lowpass_hz']) if d.get('lowpass_hz') is not None else None,notch_hz=float(d['notch_hz']) if d.get('notch_hz') is not None else None); return jsonify({'signal':y.tolist()})
    except (KeyError,TypeError,ValueError,FilterConfigurationError) as e:return jsonify({'error':str(e)}),400

@app.post('/api/detect/r-peaks')
def detect_r_peaks():
    d=_json()
    try:
        fs=float(d['sampling_rate_hz']); y=np.asarray(d['signal'],float); dm=float(d.get('min_distance_ms',250)); pf=float(d.get('prominence_factor',.5))
        if y.ndim!=1 or len(y)<8 or not np.isfinite(y).all(): raise ValueError('signal must be one-dimensional, contain at least eight finite samples, and have no NaN or infinite values')
        if not np.isfinite(fs) or fs<=0 or dm<=0 or pf<0: raise ValueError('invalid peak detector settings')
        z=y-np.median(y); scale=float(np.std(z)) or 1.; p,props=sps.find_peaks(z,distance=max(1,int(round(fs*dm/1000))),prominence=scale*pf)
        return jsonify({'peaks':p.astype(int).tolist(),'prominences':props.get('prominences',np.zeros(len(p))).tolist()})
    except (KeyError,TypeError,ValueError) as e:return jsonify({'error':str(e)}),400

@app.post('/api/beats')
@app.post('/api/segment')
def beats():
    d=_json()
    try:
        t=np.asarray(d['time'],float)
        if t.ndim!=1 or len(t)<2 or not np.isfinite(t).all() or np.any(np.diff(t)<=0): raise ValueError('time must be one-dimensional, finite, and strictly increasing')
        p=np.asarray(d.get('peaks') or [],int)
        if len(p)==0 and d.get('signal') is not None:
            fs=float(d['sampling_rate_hz']); y=np.asarray(d['signal'],float)
            if y.ndim!=1 or len(y)!=len(t) or not np.isfinite(y).all(): raise ValueError('signal must match time length and contain only finite samples')
            z=y-np.median(y); scale=float(np.std(z)) or 1.; p,_=sps.find_peaks(z,distance=max(1,int(round(fs*.25))),prominence=scale*.5)
        b=segment_beats(t,p,float(d.get('pre_s',.35)),float(d.get('post_s',.55))); return jsonify({'beats':[x.to_dict() for x in b],'peaks':p.tolist(),'n_beats':len(b)})
    except (KeyError,TypeError,ValueError) as e:return jsonify({'error':str(e)}),400

@app.post('/api/phenotype')
def phenotype():
    try:
        d=_json(); ph=beat_phenotypes(np.asarray(d['time'],float),np.asarray(d['signal'],float),np.asarray(d['r_indices'],int)); return jsonify({'beats':ph,'summary':summary_statistics(ph)})
    except (KeyError,TypeError,ValueError) as e:return jsonify({'error':str(e)}),400
@app.post('/api/statistics/compare')
def stats_compare():
    try:
        d=_json(); return jsonify(compare_groups(d['group_a'],d['group_b'],d.get('unit_ids_a'),d.get('unit_ids_b')))
    except (KeyError,TypeError,ValueError) as e:return jsonify({'error':str(e)}),400
@app.post('/api/statistics/fdr')
def stats_fdr():
    try:return jsonify({'adjusted_p':benjamini_hochberg(_json()['p_values'])})
    except (KeyError,TypeError,ValueError) as e:return jsonify({'error':str(e)}),400
@app.post('/api/benchmark')
def benchmark():
    try:
        d=_json(); return jsonify(benchmark_models(np.asarray(d['X'],float),np.asarray(d['y']),np.asarray(d['groups']),int(d.get('folds',5))))
    except (KeyError,TypeError,ValueError) as e:return jsonify({'error':str(e)}),400
@app.post('/api/ml/train')
def ml_train():
    try:
        d=_json(); m,metrics=train_classifier(np.asarray(d['signal'],float),float(d['sampling_rate_hz']),np.asarray(d['peaks'],int),list(d.get('annotations',[])),time=np.asarray(d['time'],float) if d.get('time') is not None else None); return jsonify({'trained':True,'metrics':metrics,'model':m.__class__.__name__})
    except (KeyError,TypeError,ValueError) as e:return jsonify({'trained':False,'error':str(e)}),400
@app.post('/api/ml/suggest')
def ml_suggest():
    try:
        d=_json(); y=np.asarray(d['signal'],float); fs=float(d['sampling_rate_hz']); p=np.asarray(d['peaks'],int); t=np.asarray(d['time'],float) if d.get('time') is not None else None; annotations=list(d.get('annotations',[])); m,metrics=train_classifier(y,fs,p,annotations,time=t); return jsonify({'trained':True,'metrics':metrics,'suggestions':rank_uncertain(y,fs,p,m,int(d.get('top_n',20)),time=t,annotations=annotations,min_spacing_s=float(d.get('min_spacing_s',0.25)))})
    except (KeyError,TypeError,ValueError) as e:return jsonify({'trained':False,'error':str(e)}),400
@app.get('/api/project')
def project_get():
    try:return jsonify(_project_store(request.args.get('name','default')).load().to_dict())
    except ValueError as e:return jsonify({'error':str(e)}),400
@app.post('/api/project/recording')
def project_recording():
    try:
        d=_json(); s=_project_store(d.get('project','default')); r=RecordingRef(recording_id=str(d['recording_id']),subject_id=str(d['subject_id']),group=str(d.get('group','')),visit=str(d.get('visit','')),source_path=str(d.get('source_path','')),format=str(d.get('format','')),sampling_rate_hz=d.get('sampling_rate_hz'),duration_s=d.get('duration_s'),channels=list(d.get('channels',[])),metadata=dict(d.get('metadata',{}))); return jsonify(s.add_recording(r).to_dict())
    except (KeyError,TypeError,ValueError) as e:return jsonify({'error':str(e)}),400
@app.get('/<path:path>')
def static_file(path):return send_from_directory(str(WEB),path)
if __name__=='__main__':app.run(host=os.getenv('ELECTROTRACE_HOST','127.0.0.1'),port=int(os.getenv('ELECTROTRACE_PORT','5000')),debug=False)
