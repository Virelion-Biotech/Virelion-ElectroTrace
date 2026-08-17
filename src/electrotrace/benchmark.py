"""Leakage-safe model benchmarking with subject-level splits."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GroupKFold
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
    if len(X) != len(y) or len(y) != len(groups):
        raise ValueError("X, y, and groups must have equal length")
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("Need at least two subjects/groups for leakage-safe benchmarking")
    n_splits = min(int(folds), len(unique_groups))
    if n_splits < 2:
        raise ValueError("Need at least two cross-validation folds")

    models = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1),
    }
    splitter = GroupKFold(n_splits=n_splits)
    result: dict[str, object] = {"n_samples": int(len(y)), "n_subjects": int(len(unique_groups)), "folds": n_splits, "models": {}}
    for name, model in models.items():
        metrics: list[FoldMetrics] = []
        all_true, all_pred = [], []
        confusion = None
        labels = np.unique(y)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
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
            all_true.extend(y[test_idx]); all_pred.extend(pred)
            cm = confusion_matrix(y[test_idx], pred, labels=labels)
            confusion = cm if confusion is None else confusion + cm
        result["models"][name] = {
            "folds": [m.to_dict() for m in metrics],
            "mean": {k: float(np.nanmean([getattr(m, k) if getattr(m, k) is not None else np.nan for m in metrics])) for k in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "roc_auc")},
            "confusion_matrix": confusion.tolist() if confusion is not None else [],
            "labels": [str(x) for x in labels],
        }
    return result
