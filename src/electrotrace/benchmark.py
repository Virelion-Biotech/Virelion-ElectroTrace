"""Leakage-safe model benchmarking with subject-level, stratified splits."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import t as student_t
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class FoldMetrics:
    fold: int
    n_train: int
    n_test: int
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    roc_auc: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def _model_classes(model) -> np.ndarray:
    classes = getattr(model, "classes_", None)
    if classes is not None:
        return np.asarray(classes)
    steps = getattr(model, "steps", None)
    if steps:
        estimator = steps[-1][1]
        classes = getattr(estimator, "classes_", None)
        if classes is not None:
            return np.asarray(classes)
    raise ValueError("trained model does not expose class labels")


def _auc(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> float | None:
    try:
        if len(classes) == 2:
            return float(roc_auc_score(y_true, proba[:, 1]))
        return float(roc_auc_score(y_true, proba, multi_class="ovr", labels=classes))
    except ValueError:
        return None


def _summary(values: list[float | None]) -> dict[str, float | None]:
    """Describe cross-validation fold variability.

    The t-based interval is explicitly descriptive across CV folds. Because CV
    training sets overlap, it must not be interpreted as an independent-sample
    confidence interval for future-subject performance.
    """
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if x.size == 0:
        return {"n": 0, "mean": None, "std": None, "ci95_low": None, "ci95_high": None}
    mean = float(np.mean(x))
    if x.size == 1:
        return {"n": 1, "mean": mean, "std": None, "ci95_low": None, "ci95_high": None}
    sd = float(np.std(x, ddof=1))
    critical = float(student_t.ppf(0.975, df=x.size - 1))
    margin = critical * sd / np.sqrt(x.size)
    return {"n": int(x.size), "mean": mean, "std": sd, "ci95_low": mean - margin, "ci95_high": mean + margin}


def benchmark_models(X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int = 5, seed: int = 42) -> dict:
    """Benchmark baseline classifiers using leakage-safe group-stratified folds.

    ``groups`` are the experimental units for splitting. Samples from one group
    never occur in both train and test within a fold.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    groups = np.asarray(groups)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional feature matrix")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values")
    if len(X) != len(y) or len(y) != len(groups):
        raise ValueError("X, y, and groups must have equal length")
    labels = np.unique(y)
    unique_groups = np.unique(groups)
    if len(labels) < 2:
        raise ValueError("Need at least two outcome classes for classification benchmarking")
    if len(unique_groups) < 2:
        raise ValueError("Need at least two subjects/groups for leakage-safe benchmarking")
    requested_folds = int(folds)
    if requested_folds < 2:
        raise ValueError("folds must be at least 2")
    groups_per_class = {str(cls): len(np.unique(groups[y == cls])) for cls in labels}
    max_folds = min(len(unique_groups), min(groups_per_class.values()))
    n_splits = min(requested_folds, max_folds)
    if n_splits < 2:
        raise ValueError(f"Need at least two distinct subjects in every class; observed {groups_per_class}")

    models = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1),
    }
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    result: dict[str, object] = {
        "n_samples": int(len(y)),
        "n_subjects": int(len(unique_groups)),
        "experimental_unit": "group",
        "folds": n_splits,
        "splitter": "StratifiedGroupKFold",
        "seed": int(seed),
        "groups_per_class": groups_per_class,
        "models": {},
    }
    metric_names = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "roc_auc")
    for name, model in models.items():
        metrics: list[FoldMetrics] = []
        confusion = None
        test_groups_per_fold: list[int] = []
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
            if len(np.unique(y[train_idx])) < 2:
                raise ValueError(f"Fold {fold} training set contains fewer than two outcome classes")
            if set(groups[train_idx]).intersection(set(groups[test_idx])):
                raise ValueError(f"Fold {fold} leaks experimental units between train and test")
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            proba = model.predict_proba(X[test_idx])
            classes = _model_classes(model)
            metrics.append(FoldMetrics(
                fold=fold,
                n_train=len(train_idx), n_test=len(test_idx),
                accuracy=float(accuracy_score(y[test_idx], pred)),
                balanced_accuracy=float(balanced_accuracy_score(y[test_idx], pred)),
                macro_f1=float(f1_score(y[test_idx], pred, average="macro", zero_division=0)),
                weighted_f1=float(f1_score(y[test_idx], pred, average="weighted", zero_division=0)),
                roc_auc=_auc(y[test_idx], proba, classes),
            ))
            test_groups_per_fold.append(int(len(np.unique(groups[test_idx]))))
            cm = confusion_matrix(y[test_idx], pred, labels=labels)
            confusion = cm if confusion is None else confusion + cm
        fold_values = {k: [getattr(m, k) for m in metrics] for k in metric_names}
        summaries = {k: _summary(v) for k, v in fold_values.items()}
        result["models"][name] = {
            "folds": [m.to_dict() for m in metrics],
            "summary": summaries,
            "summary_interval_interpretation": "descriptive_t_interval_across_CV_folds; not an independent-subject generalization CI",
            "mean": {k: s["mean"] for k, s in summaries.items()},
            "confusion_matrix": confusion.tolist() if confusion is not None else [],
            "labels": [str(x) for x in labels],
            "test_groups_per_fold": test_groups_per_fold,
        }
    return result
