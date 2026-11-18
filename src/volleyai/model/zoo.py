"""Catalogue des modèles candidats.

Tous prennent la même table de features (une ligne = une manchette) et sont
comparés avec le même protocole. `complexity` sert à départager : à
performance quasi égale, on garde le modèle le plus simple (plus robuste avec
peu de données, plus facile à expliquer).

Ajouter un modèle = ajouter une entrée ici, rien d'autre à changer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class Candidate:
    name: str
    complexity: int
    description: str
    build: Callable[[], Pipeline]


def _impute():
    return SimpleImputer(strategy="median", keep_empty_features=True)


CANDIDATES: dict[str, Candidate] = {c.name: c for c in [
    Candidate("logreg", 1, "Régression logistique régularisée (linéaire, coefficients lisibles)",
              lambda: Pipeline([("impute", _impute()), ("scale", StandardScaler()),
                                ("clf", LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000))])),
    Candidate("svm_rbf", 2, "SVM à noyau RBF (frontière non linéaire, données peu nombreuses)",
              lambda: Pipeline([("impute", _impute()), ("scale", StandardScaler()),
                                ("clf", CalibratedClassifierCV(
                                    SVC(C=1.0, kernel="rbf", class_weight="balanced", random_state=0),
                                    method="sigmoid", cv=3, ensemble=False))])),
    Candidate("random_forest", 3, "Forêt aléatoire peu profonde (interactions entre mesures)",
              lambda: Pipeline([("impute", _impute()),
                                ("clf", RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=2,
                                                               class_weight="balanced", random_state=0))])),
    Candidate("hist_gb", 4, "Gradient boosting (gère les mesures manquantes nativement)",
              lambda: Pipeline([("clf", HistGradientBoostingClassifier(
                  max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=3,
                  class_weight="balanced", random_state=0))])),
]}
