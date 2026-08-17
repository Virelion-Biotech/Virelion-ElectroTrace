"""Leakage-safe model benchmarking with subject-level, stratified splits."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
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


def _auc(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> float | None:
    try:
        if len(classes) == 2:
            return float(roc_auc_score(y_true, proba[:, 1]))
        return float(roc_auc_score(y_true, proba, multi_class="ovr", labels=classes))
    except ValueError:
        return None


def benchmark_models(X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int = 5, seed: int = 42) -> dict:
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
    # StratifiedGroupKFold needs enough distinct subjects in every class, not merely enough samples.
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
    result: dict[str, object] = {"n_samples": int(len(y)), "n_subjects": int(len(unique_groups)), "folds": n_splits, "splitter": "StratifiedGroupKFold", "groups_per_class": groups_per_class, "models": {}}
    for name, model in models.items():
        metrics: list[FoldMetrics] = []
        confusion = None
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
            if len(np.unique(y[train_idx])) < 2:
                raise ValueError(f"Fold {fold} training set contains fewer than two outcome classes")
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            proba = model.predict_proba(X[test_idx])
            metrics.append(FoldMetrics(
                fold=fold,
                n_train=len(train_idx), n_test=len(test_idx),
                accuracy=float(accuracy_score(y[test_idx], pred)),
                balanced_accuracy=float(balanced_accuracy_score(y[test_idx], pred)),
                macro_f1=float(f1_score(y[test_idx], pred, average="macro", zero_division=0)),
                weighted_f1=float(f1_score(y[test_idx], pred, average="weighted", zero_division=0)),
                roc_auc=_auc(y[test_idx], proba, model.classes_),
            ))
            cm = confusion_matrix(y[test_idx], pred, labels=labels)
            confusion = cm if confusion is None else confusion + cm
        result["models"][name] = {
            "folds": [m.to_dict() for m in metrics],
            "mean": {k: float(np.nanmean([getattr(m, k) if getattr(m, k) is not None else np.nan for m in metrics])) for k in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "roc_auc")},
            "confusion_matrix": confusion.tolist() if confusion is not None else [],
            "labels": [str(x) for x in labels],
        }
    return result
