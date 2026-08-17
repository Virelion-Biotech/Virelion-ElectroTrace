import numpy as np
import pytest

from electrotrace.benchmark import benchmark_models
from electrotrace.phenotype import beat_phenotypes, summary_statistics
from electrotrace.statistics import benjamini_hochberg, compare_groups


def test_phenotype_summary():
    time = np.arange(2500) / 500
    signal = np.sin(2 * np.pi * 1 * time)
    peaks = np.array([250, 750, 1250, 1750, 2250])
    beats = beat_phenotypes(time, signal, peaks)
    summary = summary_statistics(beats)
    assert summary["n_beats"] == 5
    assert summary["heart_rate_bpm"]["median"] == pytest.approx(60.0)


def test_statistics_and_fdr():
    result = compare_groups([1, 2, 3, 4], [5, 6, 7, 8])
    assert result["n_a"] == 4 and result["n_b"] == 4
    adjusted = benjamini_hochberg([0.01, 0.04, 0.2])
    assert adjusted[0] <= adjusted[1] <= adjusted[2]


def test_benchmark_requires_multiple_subjects():
    X = np.random.default_rng(1).normal(size=(10, 4))
    y = np.array([0, 1] * 5)
    with pytest.raises(ValueError):
        benchmark_models(X, y, np.ones(10), folds=2)


def test_benchmark_subject_level_split():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(24, 4))
    y = np.array([0, 1] * 12)
    groups = np.repeat(np.arange(6), 4)
    result = benchmark_models(X, y, groups, folds=3)
    assert result["n_subjects"] == 6
    assert "random_forest" in result["models"]
    assert len(result["models"]["random_forest"]["folds"]) == 3
