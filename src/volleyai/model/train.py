"""Entraînement du classifieur réussie / ratée sur les features.

Avec peu de vidéos, un modèle simple et régularisé (régression logistique)
est plus fiable qu'un réseau de neurones : il apprend des seuils sur des
mesures interprétables et ses coefficients disent quelles features comptent.

Validation croisée groupée : plusieurs clips peuvent venir de la même vidéo
source (ex. 7 fichiers "IMG_6347.mov") ; on les garde dans le même pli pour
ne pas surestimer la performance.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from volleyai.features.manchette import FEATURE_COLUMNS

MODEL_NAME = "manchette_clf.joblib"


def make_pipeline(C: float = 0.5) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=2000)),
    ])


def _groups(df: pd.DataFrame) -> np.ndarray:
    # même nom de fichier = même vidéo source probable
    return df["name"].str.lower().str.replace(r"\.[a-z0-9]+$", "", regex=True).to_numpy()


def cross_validate(df: pd.DataFrame, C: float = 0.5) -> dict:
    X, y, g = df[FEATURE_COLUMNS].to_numpy(float), df["label"].to_numpy(int), _groups(df)
    n_min = int(min((y == 0).sum(), (y == 1).sum()))
    if n_min < 2:
        return {"warning": "Pas assez d'exemples de chaque classe pour valider (min 2)."}
    n_groups = len(np.unique(g))
    splitter = (StratifiedGroupKFold(n_splits=min(5, n_min), shuffle=True, random_state=0)
                if n_min >= 3 else LeaveOneGroupOut())
    proba = np.full(len(y), np.nan)
    for tr, te in splitter.split(X, y, g):
        if len(np.unique(y[tr])) < 2:
            continue
        m = make_pipeline(C).fit(X[tr], y[tr])
        proba[te] = m.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(proba)
    pred = (proba[ok] >= 0.5).astype(int)
    res = {
        "n_samples": int(len(y)), "n_groups": int(n_groups), "n_evaluated": int(ok.sum()),
        "balanced_accuracy": round(float(balanced_accuracy_score(y[ok], pred)), 3),
        "confusion_matrix [[TN,FP],[FN,TP]]": confusion_matrix(y[ok], pred, labels=[0, 1]).tolist(),
    }
    if len(np.unique(y[ok])) == 2:
        res["roc_auc"] = round(float(roc_auc_score(y[ok], proba[ok])), 3)
    return res


def train(features_csv: str | Path, models_dir: str | Path, C: float = 0.5) -> dict:
    df = pd.read_csv(features_csv)
    df = df[df["label"].isin([0, 1])].copy()
    if df.empty:
        raise ValueError("Aucune vidéo labellisée dans la table de features.")

    cv = cross_validate(df, C)
    model = make_pipeline(C).fit(df[FEATURE_COLUMNS].to_numpy(float), df["label"].to_numpy(int))
    coefs = dict(zip(FEATURE_COLUMNS, model.named_steps["clf"].coef_[0].round(3).tolist()))
    coefs = dict(sorted(coefs.items(), key=lambda kv: -abs(kv[1])))

    meta = {"features": FEATURE_COLUMNS, "C": C, "cv": cv, "coefficients": coefs,
            "class_balance": df["label"].value_counts().to_dict()}
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "meta": meta}, models_dir / MODEL_NAME)
    (models_dir / "manchette_clf_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    _log_mlflow(meta, model)
    return meta


def _log_mlflow(meta: dict, model) -> None:
    """Journalise dans MLflow si disponible (local ou Databricks)."""
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        return
    mlflow.set_experiment("volleyai-manchette")
    with mlflow.start_run():
        mlflow.log_param("C", meta["C"])
        mlflow.log_params({f"n_class_{k}": v for k, v in meta["class_balance"].items()})
        for k, v in meta["cv"].items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)
        mlflow.log_dict(meta, "meta.json")
        mlflow.sklearn.log_model(model, "model")


def load_model(models_dir: str | Path) -> dict | None:
    p = Path(models_dir) / MODEL_NAME
    return joblib.load(p) if p.exists() else None


def predict_proba(bundle: dict, features: dict) -> float:
    x = np.array([[features.get(c, np.nan) for c in bundle["meta"]["features"]]], dtype=float)
    return float(bundle["model"].predict_proba(x)[0, 1])
