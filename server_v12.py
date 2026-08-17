"""ElectroTrace research API server v1.2."""
from __future__ import annotations
import os,sys,tempfile,uuid,zipfile
from pathlib import Path
ROOT=os.path.dirname(os.path.abspath(__file__)); SRC=os.path.join(ROOT,'src')
if SRC not in sys.path: sys.path.insert(0,SRC)
import numpy as np
from flask import Flask,jsonify,request,send_from_directory
from scipy import signal as sps
from electrotrace.io import load_csv,validate_dataframe,load_recording
from electrotrace.signal import apply_pipeline,FilterConfigurationError
from electrotrace.beats import segment_beats
from electrotrace.phenotype import beat_phenotypes,summary_statistics
from electrotrace.statistics import compare_groups,benjamini_hochberg
from electrotrace.benchmark import benchmark_models
from electrotrace.project_store import ProjectStore,RecordingRef
from electrotrace.ml import train_classifier,rank_uncertain
WEB=os.path.join(ROOT,'web'); SAMPLE_DATA=os.path.join(ROOT,'sample_data')
UPLOAD_ROOT=Path(os.getenv('ELECTROTRACE_UPLOAD_ROOT',os.path.join(tempfile.gettempdir(),'electrotrace_uploads'))); UPLOAD_ROOT.mkdir(parents=True,exist_ok=True)
PROJECT_ROOT=Path(os.getenv('ELECTROTRACE_PROJECT_ROOT',os.path.join(ROOT,'projects'))); PROJECT_ROOT.mkdir(parents=True,exist_ok=True)
app=Flask(__name__,static_folder=WEB,static_url_path='')
def _json(): return request.get_json(silent=True) or {}
def _save_upload(f):
    name=Path(f.filename or 'recording.bin'); suffix=name.suffix.lower(); token=uuid.uuid4().hex; dest=UPLOAD_ROOT/f'{token}{suffix}'; f.save(dest)
    if suffix=='.zip':
        d=UPLOAD_ROOT/token; d.mkdir()
        with zipfile.ZipFile(dest) as z:
            for m in z.infolist():
                target=(d/m.filename).resolve()
                if not str(target).startswith(str(d.resolve())): raise ValueError('Unsafe archive path')
            z.extractall(d)
        h=next(d.rglob('*.hea'),None)
        if h is None: raise ValueError('WFDB ZIP must contain .hea')
        return token,h.with_suffix('')
    return token,dest
@app.get('/')
def index(): return send_from_directory(WEB,'index.html')
@app.get('/sample_data/<path:filename>')
def sample_data(filename): return send_from_directory(SAMPLE_DATA,filename)
@app.post('/api/analyze')
def analyze():
    f=request.files.get('file')
    if f is None:return jsonify({'error':'No recording uploaded.'}),400
    try:
        df=load_csv(f.read()); r=validate_dataframe(df,request.form.get('time_col') or None); p={'valid':r.valid,'errors':r.errors,'warnings':r.warnings,'time_col':r.time_col,'signal_cols':r.signal_cols,'sampling_rate_hz':r.sampling_rate_hz,'duration_s':r.duration_s,'n_samples':r.n_samples,'format':'csv','source_format':'csv','filename':f.filename or 'recording.csv'}
        if r.valid:p['time']=df[r.time_col].astype(float).tolist();p['signals']={c:df[c].astype(float).fillna(0).tolist() for c in r.signal_cols}
        return jsonify(p)
    except Exception as e:return jsonify({'error':f'Could not read recording: {e}'}),400
@app.post('/api/recording')
def recording_upload():
    f=request.files.get('file')
    if f is None:return jsonify({'error':'No recording uploaded.'}),400
    try:
        token,path=_save_upload(f); rec=load_recording(path); return jsonify({'recording_id':token,'filename':f.filename,'format':rec.source_format,'sampling_rate_hz':rec.sampling_rate_hz,'duration_s':float(rec.time[-1]) if len(rec.time) else 0.0,'n_samples':len(rec.time),'channels':list(rec.signals.keys())})
    except Exception as e:return jsonify({'error':str(e)}),400
@app.get('/api/recording/<rid>/window')
def window(rid):
    try:
        ms=list(UPLOAD_ROOT.glob(rid))+list(UPLOAD_ROOT.glob(rid+'.*')); path=ms[0] if ms else None
        if path is None and (UPLOAD_ROOT/rid).is_dir(): path=next((x.with_suffix('') for x in (UPLOAD_ROOT/rid).rglob('*.hea')),None)
        if path is None:return jsonify({'error':'recording not found'}),404
        rec=load_recording(path); s=max(0,int(request.args.get('start',0))); e=min(len(rec.time),int(request.args.get('stop',min(s+5000,len(rec.time))))); 
        if e<=s:raise ValueError('invalid window')
        return jsonify({'start':s,'stop':e,'time':rec.time[s:e].tolist(),'signals':{c:y[s:e].tolist() for c,y in rec.signals.items()}})
    except Exception as e:return jsonify({'error':str(e)}),400
@app.post('/api/filter')
def filter_signal():
    d=_json()
    try:return jsonify({'signal':apply_pipeline(np.asarray(d['signal'],float),float(d['sampling_rate_hz']),baseline=bool(d.get('baseline',False)),highpass_hz=float(d['highpass_hz']) if d.get('highpass_hz') is not None else None,lowpass_hz=float(d['lowpass_hz']) if d.get('lowpass_hz') is not None else None,notch_hz=float(d['notch_hz']) if d.get('notch_hz') is not None else None).tolist()})
    except (KeyError,TypeError,ValueError,FilterConfigurationError) as e:return jsonify({'error':str(e)}),400
@app.post('/api/detect/r-peaks')
def rpeaks():
    d=_json()
    try:
        fs=float(d['sampling_rate_hz']); y=np.asarray(d['signal'],float); dist=max(1,int(round(fs*float(d.get('min_distance_ms',250))/1000))); z=np.nan_to_num(y-np.nanmedian(y)); scale=float(np.nanstd(z)) or 1.0; p,props=sps.find_peaks(z,distance=dist,prominence=scale*float(d.get('prominence_factor',.5))); return jsonify({'peaks':p.tolist(),'prominences':props.get('prominences',np.zeros(len(p))).tolist()})
    except (KeyError,TypeError,ValueError) as e:return jsonify({'error':str(e)}),400
@app.post('/api/beats')
@app.post('/api/segment')
def beats():
    d=_json()
    try:
        t=np.asarray(d['time'],float); p=np.asarray(d.get('peaks') or [],int)
        if len(p)==0 and d.get('signal') is not None:
            fs=float(d['sampling_rate_hz']); y=np.asarray(d['signal'],float); z=np.nan_to_num(y-np.nanmedian(y)); scale=float(np.nanstd(z)) or 1.0; p,_=sps.find_peaks(z,distance=max(1,int(round(fs*.25))),prominence=scale*.5)
        b=segment_beats(t,p,float(d.get('pre_s',.35)),float(d.get('post_s',.55))); return jsonify({'beats':[x.to_dict() for x in b],'peaks':p.tolist(),'n_beats':len(b)})
    except (KeyError,TypeError,ValueError) as e:return jsonify({'error':str(e)}),400
@app.post('/api/phenotype')
def phenotype():
    d=_json()
    try:
        ph=beat_phenotypes(np.asarray(d['time'],float),np.asarray(d['signal'],float),np.asarray(d['r_indices'],int)); return jsonify({'beats':ph,'summary':summary_statistics(ph)})
    except (KeyError,TypeError,ValueError) as e:return jsonify({'error':str(e)}),400
@app.post('/api/statistics/compare')
def compare():
    d=_json()
    try:return jsonify(compare_groups(d['group_a'],d['group_b']))
    except (KeyError,TypeError,ValueError) as e:return jsonify({'error':str(e)}),400
@app.post('/api/statistics/fdr')
def fdr():
    d=_json()
    try:return jsonify({'adjusted_p':benjamini_hochberg(d['p_values'])})
    except (KeyError,TypeError,ValueError) as e:return jsonify({'error':str(e)}),400
@app.post('/api/benchmark')
def benchmark():
    d=_json()
    try:return jsonify(benchmark_models(np.asarray(d['X'],float),np.asarray(d['y']),np.asarray(d['groups']),int(d.get('folds',5))))
    except (KeyError,TypeError,ValueError) as e:return jsonify({'error':str(e)}),400
@app.post('/api/ml/train')
def mltrain():
    d=_json()
    try:m,metrics=train_classifier(np.asarray(d['signal'],float),float(d['sampling_rate_hz']),np.asarray(d['peaks'],int),list(d.get('annotations',[])));return jsonify({'trained':True,'metrics':metrics,'model':m.__class__.__name__})
    except (KeyError,TypeError,ValueError) as e:return jsonify({'trained':False,'error':str(e)}),400
@app.post('/api/ml/suggest')
def mlsuggest():
    d=_json()
    try:
        y=np.asarray(d['signal'],float); fs=float(d['sampling_rate_hz']); p=np.asarray(d['peaks'],int); m,metrics=train_classifier(y,fs,p,list(d.get('annotations',[]))); return jsonify({'trained':True,'metrics':metrics,'suggestions':rank_uncertain(y,fs,p,m,int(d.get('top_n',20)))})
    except (KeyError,TypeError,ValueError) as e:return jsonify({'trained':False,'error':str(e)}),400
@app.get('/api/project')
def project_get():return jsonify(ProjectStore(PROJECT_ROOT/request.args.get('name','default')).load().to_dict())
@app.post('/api/project/recording')
def project_recording():
    d=_json()
    try:
        s=ProjectStore(PROJECT_ROOT/str(d.get('project','default'))); r=RecordingRef(recording_id=str(d['recording_id']),subject_id=str(d['subject_id']),group=str(d.get('group','')),visit=str(d.get('visit','')),source_path=str(d.get('source_path','')),format=str(d.get('format','')),sampling_rate_hz=d.get('sampling_rate_hz'),duration_s=d.get('duration_s'),channels=list(d.get('channels',[])),metadata=dict(d.get('metadata',{}))); return jsonify(s.add_recording(r).to_dict())
    except (KeyError,TypeError,ValueError) as e:return jsonify({'error':str(e)}),400
@app.get('/<path:path>')
def static_file(path):return send_from_directory(WEB,path)
if __name__=='__main__':app.run(host=os.getenv('ELECTROTRACE_HOST','127.0.0.1'),port=int(os.getenv('ELECTROTRACE_PORT','5000')),debug=False)
