"""Genotype-linked ECG phenotype metadata without signal-to-genotype inference."""
from __future__ import annotations
from typing import Any, Mapping
import math

PROFILE_VERSION="electrotrace-genotype-ep-phenotype/1.0.0"
SUPPORTED_GENES={"PKP2","PLN","BAG3","LAMP2","CASQ2"}
SOURCE_TYPES={"peer_reviewed","clinical_registry","regulator","company_release","conference_abstract","other"}

def _text(v): return isinstance(v,str) and bool(v.strip())

def _validate_sources(sources:Any)->list[str]:
    errors=[]
    if not isinstance(sources,list) or not sources: return ["source_provenance must be a non-empty array"]
    for i,s in enumerate(sources):
        if not isinstance(s,dict): errors.append(f"source_provenance[{i}] must be an object"); continue
        for k in ("source_identifier","source_type","source_url","source_sha256"):
            if not _text(s.get(k)): errors.append(f"source_provenance[{i}].{k} is required")
        if s.get("source_type") not in SOURCE_TYPES: errors.append(f"source_provenance[{i}].source_type is unsupported")
        digest=s.get("source_sha256")
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
            errors.append(f"source_provenance[{i}].source_sha256 must be SHA-256")
    return errors

def validate_genotype_context(context:Mapping[str,Any]|None)->list[str]:
    if context is None: return []
    errors=[]
    status=context.get("status","unknown")
    if status not in {"unknown","reported","curated"}: errors.append("genotype status must be unknown, reported or curated")
    if context.get("inferred_from_ecg") is True: errors.append("genotype must never be inferred from ECG")
    variants=context.get("variants",[])
    if not isinstance(variants,list): errors.append("variants must be an array")
    else:
        for i,v in enumerate(variants):
            if not isinstance(v,dict) or not _text(v.get("gene_symbol")): errors.append(f"variants[{i}] requires gene_symbol"); continue
            if v["gene_symbol"].upper() not in SUPPORTED_GENES: errors.append(f"variants[{i}] gene_symbol is not in the supported curation set")
            if status!="unknown" and not _text(v.get("variant_nomenclature")): errors.append(f"variants[{i}] variant_nomenclature must be supplied or explicitly unknown")
    if status in {"reported","curated"}:
        if not context.get("source_record_ids"): errors.append("reported/curated genotype requires source_record_ids")
        errors.extend(_validate_sources(context.get("source_provenance")))
    return errors

def _hrv(rr:Any)->dict[str,Any]:
    if not isinstance(rr,list) or len(rr)<3:
        return {"status":"not_estimable","reason":"at_least_three_rr_intervals_required","method":"time-domain from supplied/detected R-peak intervals"}
    values=[float(x) for x in rr]
    if any(not math.isfinite(x) or x<=0 for x in values):
        return {"status":"not_estimable","reason":"RR intervals must be finite and positive"}
    diffs=[values[i+1]-values[i] for i in range(len(values)-1)]
    sdnn=math.sqrt(sum((x-sum(values)/len(values))**2 for x in values)/(len(values)-1))
    rmssd=math.sqrt(sum(x*x for x in diffs)/len(diffs))
    return {"status":"descriptive_short_recording","n_intervals":len(values),"mean_rr_ms":1000*sum(values)/len(values),"sdnn_ms":1000*sdnn,"rmssd_ms":1000*rmssd,"method":"SDNN sample standard deviation and RMSSD of successive RR differences","clinical_interpretation":"not_established"}

def _qt_feature(qt:Mapping[str,Any]|None)->dict[str,Any]:
    if qt is None: return {"status":"not_measured","qt_ms":None,"qtc_ms":None,"method":None,"reason":"no externally supplied/adjudicated QT interval"}
    required=("qt_ms","rr_interval_s","method","lead","adjudication_status")
    missing=[k for k in required if k not in qt]
    if missing: raise ValueError("QT measurement missing fields: "+", ".join(missing))
    qt_ms=float(qt["qt_ms"]); rr=float(qt["rr_interval_s"])
    if not math.isfinite(qt_ms) or qt_ms<=0 or not math.isfinite(rr) or rr<=0: raise ValueError("QT and RR must be finite and positive")
    method=str(qt["method"]).lower()
    if method=="bazett": qtc=qt_ms/math.sqrt(rr)
    elif method=="fridericia": qtc=qt_ms/(rr**(1/3))
    else: raise ValueError("QTc method must be bazett or fridericia")
    return {"status":"externally_supplied_measurement","qt_ms":qt_ms,"rr_interval_s":rr,"qtc_ms":qtc,"method":method,"lead":qt["lead"],"adjudication_status":qt["adjudication_status"],"source_record_ids":list(qt.get("source_record_ids",[])),"clinical_interpretation":"no threshold or diagnosis applied"}

def build_genotype_phenotype_profile(ecg_bundle:dict[str,Any],*,genotype_context:Mapping[str,Any]|None=None,phenotype_label:Mapping[str,Any]|None=None,subject_id:str|None=None,study_id:str|None=None,split_group_id:str|None=None,rhythm_context:Mapping[str,Any]|None=None,qt_measurement:Mapping[str,Any]|None=None,synthetic:Mapping[str,Any]|None=None)->dict[str,Any]:
    """Attach supplied genotype/labels to an ECG bundle and expose descriptive feature families."""
    if not isinstance(ecg_bundle,dict) or ecg_bundle.get("schema_version")!="electrotrace-ecg-measurement-v2":
        raise ValueError("a validated electrotrace-ecg-measurement-v2 bundle is required")
    errors=validate_genotype_context(genotype_context)
    if errors: raise ValueError("; ".join(errors))
    genotype=dict(genotype_context or {"status":"unknown","variants":[]})
    genotype.setdefault("status","unknown"); genotype.setdefault("variants",[])
    label=dict(phenotype_label or {"status":"unknown","label":None})
    if label.get("status") in {"observed","curated"} and not _text(label.get("label")): raise ValueError("observed/curated phenotype label requires a non-empty label")
    if label.get("status")=="synthetic" and (synthetic is None or synthetic.get("seed") is None or not _text(synthetic.get("generator_version"))): raise ValueError("synthetic phenotype label requires synthetic seed and generator_version")
    if label.get("status") not in {"unknown","observed","curated","synthetic"}: raise ValueError("phenotype label status is unsupported")
    if label.get("status") in {"observed","curated"}:
        if not label.get("source_record_ids"): raise ValueError("observed/curated phenotype label requires source_record_ids")
        errs=_validate_sources(label.get("source_provenance"))
        if errs: raise ValueError("; ".join(errs))
    if synthetic is not None and (synthetic.get("seed") is None or not _text(synthetic.get("generator_version"))):
        raise ValueError("synthetic profile requires seed and generator_version")
    beat_analysis=ecg_bundle.get("beat_analysis",{})
    beats=beat_analysis.get("beats",[]) if isinstance(beat_analysis,dict) else []
    ectopic=sum(1 for b in beats if b.get("beat_annotation",{}).get("ectopic") is True)
    paced=sum(1 for b in beats if b.get("beat_annotation",{}).get("paced") is True)
    unknown=sum(1 for b in beats if b.get("beat_annotation",{}).get("ectopic") is None and b.get("beat_annotation",{}).get("paced") is None)
    n=len(beats)
    qrs=ecg_bundle.get("morphology",{}).get("qrs",{})
    morphology=ecg_bundle.get("morphology",{})
    provenance=ecg_bundle.get("provenance",{})
    source_hash=ecg_bundle.get("raw_measurements",{}).get("source_reference",{}).get("sha256")
    return {
      "schema_version":"electrotrace-genotype-ep-phenotype-v1","profile_version":PROFILE_VERSION,
      "evidence_state":"synthetic" if synthetic is not None else "observed",
      "genotype_context":genotype,"phenotype_label":label,
      "subject_linkage":{"subject_id":subject_id,"study_id":study_id,"split_group_id":split_group_id or subject_id,"status":"explicit" if subject_id else "unknown_no_subject_inference"},
      "acquisition":{"sampling_frequency_hz":ecg_bundle.get("acquisition",{}).get("sampling_frequency_hz"),"voltage_units":ecg_bundle.get("acquisition",{}).get("physical_voltage_unit_by_lead"),"lead_order":ecg_bundle.get("acquisition",{}).get("lead_names_and_order"),"device_metadata":ecg_bundle.get("acquisition",{}).get("device_metadata"),"filter_metadata":ecg_bundle.get("acquisition",{}).get("filter_metadata"),"source_sha256":source_hash},
      "rhythm_context":dict(rhythm_context or {"status":"not_reported"}),
      "feature_families":{
        "conduction":{"status":"measured_or_partially_measured" if qrs else "not_estimable","qrs":qrs},
        "qrs_morphology":{"status":"measured" if morphology.get("robust_template") else "not_estimable","robust_template":morphology.get("robust_template"),"beat_to_beat_variability_by_lead":morphology.get("beat_to_beat_variability_by_lead")},
        "repolarization":{"status":"exploratory_proxy_only" if morphology.get("p_st_t_measurements") else "not_measured","measurements":morphology.get("p_st_t_measurements")},
        "qt_qtc":_qt_feature(qt_measurement),
        "ectopy_pacing":{"status":"annotation_based_not_signal_inferred","n_beats":n,"adjudicated_ectopic_count":ectopic,"adjudicated_paced_count":paced,"unclassified_count":unknown,"ectopic_burden_fraction":ectopic/n if n else None},
        "heart_rate_variability":_hrv(beat_analysis.get("rr_intervals_s") if isinstance(beat_analysis,dict) else None),
        "signal_quality":{"quality":ecg_bundle.get("quality"),"uncertainty":ecg_bundle.get("measurement_uncertainty")}
      },
      "feature_provenance":{"algorithm_versions":provenance.get("algorithm_versions",{}),"detector_settings":provenance.get("detector_settings",{}),"source_sha256":source_hash,"uncertainty_status":"see_measurement_bundle"},
      "synthetic_metadata":dict(synthetic or {}),
      "safety_policy":{"genotype_never_inferred_from_ecg":True,"disease_associations_descriptive_only":True,"no_diagnostic_thresholds":True,"no_clinical_recommendations":True}
    }
