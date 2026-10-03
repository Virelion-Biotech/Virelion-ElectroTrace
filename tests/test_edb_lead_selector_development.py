import pytest

from scripts import develop_edb_label_free_lead_selector as lead


def _channel(*, qrs, prom=2.0, width=0.04, p50=0.9, retention=0.5, f1=0.8, tp=80, fp=10, fn=20):
    sens = tp / (tp + fn) if tp + fn else 0.0
    ppv = tp / (tp + fp) if tp + fp else 0.0
    return {
        "feature_medians_retained": {
            "qrs_band_fraction": qrs,
            "prominence_z": prom,
            "width_s": width,
        },
        "label_free_quality": {
            "retention_fraction": retention,
            "retained_probability": {"p50": p50},
        },
        "metrics": {
            "f1": f1,
            "sensitivity": sens,
            "positive_predictive_value": ppv,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
        },
    }


def test_label_free_scores_do_not_require_reference_fields():
    ch = _channel(qrs=0.6, prom=4.0, width=0.03, p50=0.95, retention=0.7)
    scores = lead.label_free_scores(ch)
    assert scores["retained_qrs_band_fraction"] == pytest.approx(0.6)
    assert scores["qrs_band_x_retained_probability"] == pytest.approx(0.57)
    assert scores["qrs_band_per_width"] == pytest.approx(20.0)


def test_choose_channel_uses_label_free_score_and_ties_to_channel0():
    ch0 = _channel(qrs=0.4)
    ch1 = _channel(qrs=0.7, f1=0.1)
    assert lead.choose_channel("retained_qrs_band_fraction", ch0, ch1) == 1

    tied = _channel(qrs=0.4, f1=0.99)
    assert lead.choose_channel("retained_qrs_band_fraction", ch0, tied) == 0


def test_summary_uses_selected_channels_and_counts_regressions():
    ch0a = _channel(qrs=0.2, f1=0.5, tp=50, fp=10, fn=50)
    ch1a = _channel(qrs=0.8, f1=0.9, tp=90, fp=5, fn=10)
    ch0b = _channel(qrs=0.8, f1=0.95, tp=95, fp=5, fn=5)
    ch1b = _channel(qrs=0.2, f1=0.7, tp=70, fp=10, fn=30)

    rows = [
        {
            "channel0": ch0a,
            "channel1": ch1a,
            "selectors": {"rule": 1},
        },
        {
            "channel0": ch0b,
            "channel1": ch1b,
            "selectors": {"rule": 0},
        },
    ]
    summary = lead.summarize_selected(rows, "rule")
    assert summary["records"] == 2
    assert summary["selected_channel1_records"] == 1
    assert summary["records_improved_vs_channel0"] == 1
    assert summary["records_regressed_vs_channel0"] == 0
    assert summary["true_positive"] == 185
    assert summary["false_positive"] == 10
    assert summary["false_negative"] == 15
