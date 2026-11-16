import pytest
from electrotrace.genotype_phenotype import build_genotype_phenotype_profile, validate_genotype_context

def bundle():
    return {
      "schema_version":"electrotrace-ecg-measurement-v2",
      "acquisition":{"sampling_frequency_hz":500,"physical_voltage_unit_by_lead":{"I":"mV"},"lead_names_and_order":["I"],"device_metadata":{},"filter_metadata":{}},
      "raw_measurements":{"source_reference":{"sha256":"a"*64}},
      "beat_analysis":{"rr_intervals_s":[0.8,0.82,0.78],"beats":[{"beat_annotation":{"ectopic":True,"paced":False}},{"beat_annotation":{"ectopic":None,"paced":None}}]},
      "morphology":{"qrs":{"median_duration_ms":90},"robust_template":{"status":"available"},"beat_to_beat_variability_by_lead":{"I":{}},"p_st_t_measurements":{"validation_status":"exploratory_proxy_not_independently_validated"}},
      "provenance":{"algorithm_versions":{"qrs":"1"},"detector_settings":{"name":"fixture"}},
      "quality":{"status":"provisional_pass"},"measurement_uncertainty":{"independent_adjudication_calibration":{"status":"not_run"}}
    }

def source():
    return [{"source_identifier":"PMID:123","source_type":"peer_reviewed","source_url":"https://example.org/source","source_sha256":"b"*64}]

def test_genotype_is_never_inferred_from_ecg():
    ctx={"status":"curated","variants":[{"gene_symbol":"PKP2","variant_nomenclature":"NM_001005242.3:c.2013del"}],"source_record_ids":["atlas-1"],"source_provenance":source()}
    profile=build_genotype_phenotype_profile(bundle(),genotype_context=ctx,subject_id="S1")
    assert profile["genotype_context"]["variants"][0]["gene_symbol"]=="PKP2"
    assert profile["safety_policy"]["genotype_never_inferred_from_ecg"]
    assert profile["subject_linkage"]["split_group_id"]=="S1"

def test_missing_genotype_remains_unknown():
    p=build_genotype_phenotype_profile(bundle())
    assert p["genotype_context"]["status"]=="unknown"

def test_reported_genotype_requires_source_provenance():
    with pytest.raises(ValueError,match="source_record_ids"):
        build_genotype_phenotype_profile(bundle(),genotype_context={"status":"reported","variants":[{"gene_symbol":"PLN","variant_nomenclature":"c.40_42del"}]})

def test_ecg_features_do_not_generate_genotype():
    p=build_genotype_phenotype_profile(bundle())
    assert p["genotype_context"]["variants"]==[]
    assert p["feature_families"]["conduction"]["qrs"]["median_duration_ms"]==90

def test_qtc_requires_explicit_method_and_annotation():
    p=build_genotype_phenotype_profile(bundle())
    assert p["feature_families"]["qt_qtc"]["status"]=="not_measured"
    p=build_genotype_phenotype_profile(bundle(),qt_measurement={"qt_ms":400,"rr_interval_s":0.8,"method":"fridericia","lead":"II","adjudication_status":"expert_reviewed"})
    assert p["feature_families"]["qt_qtc"]["qtc_ms"]>400

def test_ectopy_burden_uses_annotations_only():
    p=build_genotype_phenotype_profile(bundle())
    ect=p["feature_families"]["ectopy_pacing"]
    assert ect["adjudicated_ectopic_count"]==1
    assert ect["unclassified_count"]==1

def test_synthetic_profile_requires_seed_and_generator():
    with pytest.raises(ValueError,match="seed"):
        build_genotype_phenotype_profile(bundle(),synthetic={"generator_version":"test"})


def test_unsupported_gene_symbol_is_rejected():
    errors = validate_genotype_context({
        "status": "curated",
        "variants": [{"gene_symbol": "NOT_A_SUPPORTED_GENE", "variant_nomenclature": "c.1A>G"}],
        "source_record_ids": ["source-1"],
        "source_provenance": source(),
    })
    assert any("not in the supported curation set" in error for error in errors)
