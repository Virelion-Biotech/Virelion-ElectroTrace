from __future__ import annotations
import numpy as np
from scipy import signal as sps
from electrotrace.beats import segment_beats
from electrotrace.phenotype import beat_phenotypes, summary_statistics
from electrotrace.statistics import compare_groups, benjamini_hochberg
from electrotrace.benchmark import benchmark_models
from electrotrace.ml import train_classifier, rank_uncertain

def register_research_routes(app):
    @app.post('/api/segment')
    @app.post('/api/beats')
    def beats_endpoint():
        data = app.current_request.get_json(silent=True) if hasattr(app,'current_request') else {}
        time=np.asarray(data['time'],dtype=float); peaks=np.asarray(data.get('peaks') or [],dtype=int)
        if len(peaks)==0 and data.get('signal') is not None:
            fs=float(data['sampling_rate_hz']); y=np.asarray(data['signal'],dtype=float)
            distance=max(1,int(round(fs*.25))); centered=np.nan_to_num(y-np.nanmedian(y)); scale=float(np.nanstd(centered)) or 1.0
            peaks,_=sps.find_peaks(centered,distance=distance,prominence=scale*.5)
        beats=segment_beats(time,peaks,float(data.get('pre_s',.35)),float(data.get('post_s',.55)))
        return {'beats':[b.to_dict() for b in beats],'peaks':peaks.tolist(),'n_beats':len(beats)}
