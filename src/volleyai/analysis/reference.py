"""Profil de référence : "quels sont les bons paramètres d'une manchette réussie ?"

Pour chaque feature, compare la distribution des manchettes réussies et ratées :
médiane, plage p10–p90 des réussies, pouvoir discriminant (AUC), et un seuil
suggéré (indice de Youden). C'est ce rapport qui permet d'ajuster les seuils
des règles avec les données plutôt qu'à l'intuition.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from volleyai.features.manchette import FEATURE_COLUMNS, FEATURE_DESCRIPTIONS


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC = P(réussie > ratée) (Mann-Whitney), sans dépendance sklearn."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def _youden_threshold(pos: np.ndarray, neg: np.ndarray, higher_is_good: bool) -> float | None:
    vals = np.unique(np.r_[pos, neg])
    if len(vals) < 2:
        return None
    best, best_j = None, -1.0
    for t in vals:
        tpr = (pos >= t).mean() if higher_is_good else (pos <= t).mean()
        fpr = (neg >= t).mean() if higher_is_good else (neg <= t).mean()
        if tpr - fpr > best_j:
            best, best_j = float(t), tpr - fpr
    return best


def build_reference(df: pd.DataFrame, label_col: str = "label") -> dict:
    ref = {"n_reussies": int((df[label_col] == 1).sum()), "n_ratees": int((df[label_col] == 0).sum()),
           "features": {}}
    for col in FEATURE_COLUMNS:
        if col not in df:
            continue
        pos = df.loc[df[label_col] == 1, col].dropna().to_numpy(float)
        neg = df.loc[df[label_col] == 0, col].dropna().to_numpy(float)
        if len(pos) == 0:
            continue
        auc = _auc(pos, neg)
        higher_good = auc >= 0.5 if not math.isnan(auc) else True
        ref["features"][col] = {
            "description": FEATURE_DESCRIPTIONS.get(col, col),
            "reussie_median": float(np.median(pos)),
            "reussie_p10": float(np.percentile(pos, 10)),
            "reussie_p90": float(np.percentile(pos, 90)),
            "ratee_median": float(np.median(neg)) if len(neg) else None,
            "auc": None if math.isnan(auc) else round(auc, 3),
            "discriminant": None if math.isnan(auc) else round(abs(auc - 0.5) * 2, 3),
            "direction": "plus haut = mieux" if higher_good else "plus bas = mieux",
            "seuil_suggere": _youden_threshold(pos, neg, higher_good) if len(neg) else None,
            "n": [len(pos), len(neg)],
        }
    # tri par pouvoir discriminant
    ref["features"] = dict(sorted(ref["features"].items(),
                                  key=lambda kv: -(kv[1]["discriminant"] or 0)))
    return ref


def save_reference(ref: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(ref, indent=2, ensure_ascii=False), encoding="utf-8")


def load_reference(path: str | Path) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def compare_to_reference(features: dict, ref: dict, top: int = 5) -> list[dict]:
    """Features hors de la plage p10–p90 des manchettes réussies."""
    out = []
    for col, r in ref.get("features", {}).items():
        v = features.get(col)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        lo, hi = r["reussie_p10"], r["reussie_p90"]
        if lo <= v <= hi:
            continue
        span = max(hi - lo, 1e-6)
        dist = (lo - v) / span if v < lo else (v - hi) / span
        out.append({"feature": col, "description": r["description"], "valeur": round(float(v), 3),
                    "plage_reussies": [round(lo, 3), round(hi, 3)],
                    "ecart": round(float(dist), 2), "discriminant": r["discriminant"]})
    out.sort(key=lambda d: -(d["ecart"] * (d["discriminant"] or 0.1)))
    return out[:top]
