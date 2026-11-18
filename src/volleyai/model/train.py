"""Entraînement et comparaison de plusieurs modèles réussie / ratée.

Protocole identique pour tous les candidats :
- validation croisée **groupée** (les clips d'une même vidéo source restent
  ensemble) et **répétée** (plusieurs tirages) : avec peu de vidéos, un seul
  découpage donne un score très bruité, on reporte donc moyenne ± écart-type ;
- chaque candidat = un run MLflow (paramètres, métriques, modèle, importances) ;
- le meilleur est enregistré dans le registre MLflow avec l'alias
  ``challenger`` ; il ne devient ``champion`` (modèle servi par l'app) que
  s'il n'y en a pas encore, ou sur promotion explicite.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold

from volleyai.features.manchette import FEATURE_COLUMNS
from volleyai.model.zoo import CANDIDATES

METRICS = ("balanced_accuracy", "roc_auc", "f1_ratee")
# types sûrs (numpy / sklearn) que MLflow doit accepter au rechargement des modèles
SKOPS_TRUSTED = ["numpy.dtype", "sklearn.tree._tree.Tree",
                 "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
                 "sklearn.calibration._CalibratedClassifier", "sklearn.calibration._SigmoidCalibration"]


def groups_of(df: pd.DataFrame) -> np.ndarray:
    """Même nom de fichier = même vidéo source probable (ex. 7 × IMG_6347.mov)."""
    return df["name"].astype(str).str.lower().str.replace(r"\.[a-z0-9]+$", "", regex=True).to_numpy()


def _splits(y: np.ndarray, g: np.ndarray, seed: int):
    n_min = int(min((y == 0).sum(), (y == 1).sum()))
    if n_min >= 3 and len(np.unique(g)) >= 3:
        return StratifiedGroupKFold(n_splits=min(5, n_min), shuffle=True, random_state=seed).split(y, y, g)
    return LeaveOneGroupOut().split(y, y, g)


def cross_validate(build, X: pd.DataFrame, y: np.ndarray, g: np.ndarray, repeats: int = 5) -> dict:
    scores = {m: [] for m in METRICS}
    for seed in range(repeats):
        proba = np.full(len(y), np.nan)
        for tr, te in _splits(y, g, seed):
            if len(np.unique(y[tr])) < 2:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = build().fit(X.iloc[tr], y[tr])
            proba[te] = m.predict_proba(X.iloc[te])[:, 1]
        ok = ~np.isnan(proba)
        if ok.sum() == 0 or len(np.unique(y[ok])) < 2:
            continue
        pred = (proba[ok] >= 0.5).astype(int)
        scores["balanced_accuracy"].append(balanced_accuracy_score(y[ok], pred))
        scores["roc_auc"].append(roc_auc_score(y[ok], proba[ok]))
        scores["f1_ratee"].append(f1_score(y[ok], pred, pos_label=0, zero_division=0))
    out = {}
    for m, v in scores.items():
        if v:
            out[f"{m}_mean"] = round(float(np.mean(v)), 4)
            out[f"{m}_std"] = round(float(np.std(v)), 4)
    return out


def _data_hash(df: pd.DataFrame) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(df[FEATURE_COLUMNS + ["label"]], index=False).values).hexdigest()[:12]


def _pick_best(board: pd.DataFrame, metric: str, margin: float) -> str:
    """Meilleur score ; à moins de `margin` du meilleur, on prend le plus simple."""
    col = f"{metric}_mean"
    best = board[col].max()
    close = board[board[col] >= best - margin].sort_values(["complexity", col], ascending=[True, False])
    return str(close.iloc[0]["candidate"])


def train_candidates(features_csv: str | Path, cfg: dict, candidates: list[str] | None = None,
                     register: bool = True) -> dict:
    tcfg, mcfg = cfg.get("training", {}), cfg.get("mlflow", {})
    df = pd.read_csv(features_csv)
    df = df[df["label"].isin([0, 1])].reset_index(drop=True)
    if df["label"].nunique() < 2:
        raise ValueError("Il faut des manchettes réussies ET ratées pour entraîner.")
    for c in FEATURE_COLUMNS:
        if c not in df:
            df[c] = np.nan
    X, y, g = df[FEATURE_COLUMNS].astype(float), df["label"].to_numpy(int), groups_of(df)
    names = candidates or tcfg.get("candidates") or list(CANDIDATES)
    repeats = int(tcfg.get("cv_repeats", 5))
    dataset = {"n_samples": len(df), "n_reussies": int((y == 1).sum()), "n_ratees": int((y == 0).sum()),
               "n_groups": int(len(np.unique(g))), "data_hash": _data_hash(df),
               "source": str(df["source_kind"].iloc[0]) if "source_kind" in df else "videos"}

    mlflow = _mlflow(cfg)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    rows, fitted = [], {}
    parent = mlflow.start_run(run_name=f"train-{stamp}") if mlflow else None
    try:
        if mlflow:
            mlflow.log_params({f"data_{k}": v for k, v in dataset.items()})
            mlflow.log_param("cv_repeats", repeats)
        for name in names:
            cand = CANDIDATES[name]
            try:
                cv = cross_validate(cand.build, X, y, g, repeats)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = cand.build().fit(X, y)
            except Exception as e:  # un candidat qui plante ne bloque pas les autres
                print(f"[train] {name} : échec ({e})")
                continue
            imp = permutation_importance(model, X, y, n_repeats=10, random_state=0, scoring="balanced_accuracy")
            importances = dict(sorted(zip(FEATURE_COLUMNS, imp.importances_mean.round(4).tolist()),
                                      key=lambda kv: -kv[1]))
            row = {"candidate": name, "complexity": cand.complexity, **cv}
            if mlflow:
                with mlflow.start_run(run_name=name, nested=True) as run:
                    mlflow.set_tags({"candidate": name, "description": cand.description, **{
                        f"data_{k}": v for k, v in dataset.items()}})
                    mlflow.log_params({k: v for k, v in model.get_params().items()
                                       if isinstance(v, (int, float, str, bool)) and "__" in k})
                    mlflow.log_metrics(cv)
                    mlflow.log_dict(importances, "feature_importance.json")
                    info = mlflow.sklearn.log_model(model, name="model", input_example=X.head(2).astype("float64"),
                                                    skops_trusted_types=SKOPS_TRUSTED)
                    row["run_id"], row["model_uri"] = run.info.run_id, info.model_uri
            rows.append(row)
            fitted[name] = (model, importances)
            print(f"[train] {name:14s} " + "  ".join(f"{k}={v}" for k, v in cv.items() if k.endswith("mean")))

        board = pd.DataFrame(rows)
        if board.empty:
            raise RuntimeError("Aucun candidat n'a pu être entraîné.")
        metric = tcfg.get("selection_metric", "balanced_accuracy")
        best = _pick_best(board, metric, float(tcfg.get("simplicity_margin", 0.02)))
        result = {"leaderboard": board, "best": best, "dataset": dataset, "registered": None}
        if mlflow:
            mlflow.set_tag("best_candidate", best)
        if mlflow and register:
            result["registered"] = _register(mlflow, cfg, board.set_index("candidate").loc[best], dataset)
    finally:
        if parent:
            mlflow.end_run()

    # copie locale (secours si MLflow n'est pas installé) + rapport
    import joblib
    models_dir = Path(cfg["paths"]["models_dir"]); models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": fitted[best][0], "meta": {"candidate": best, "dataset": dataset,
                 "importances": fitted[best][1], **board.set_index("candidate").loc[best].to_dict()}},
                models_dir / "manchette_clf.joblib")
    reports = Path(cfg["paths"]["reports_dir"]); reports.mkdir(parents=True, exist_ok=True)
    board.to_csv(reports / "leaderboard.csv", index=False)
    return result


# --------------------------------------------------------------------------
# MLflow
# --------------------------------------------------------------------------
def _mlflow(cfg: dict):
    try:
        import mlflow
        import mlflow.sklearn  # noqa: F401
    except ImportError:
        print("[train] MLflow non installé : modèles sauvegardés en local uniquement.")
        return None
    m = cfg.get("mlflow", {})
    mlflow.set_tracking_uri(m.get("tracking_uri"))
    if m.get("registry_uri"):
        mlflow.set_registry_uri(m["registry_uri"])
    exp = m.get("experiment", "volleyai-manchette")
    uri = m.get("tracking_uri") or ""
    if uri.startswith("sqlite:") and mlflow.get_experiment_by_name(exp) is None:
        # artefacts à côté de la base, pas dans le dossier courant
        root = Path(uri.replace("sqlite:///", "")).parent
        mlflow.create_experiment(exp, artifact_location=(root / "mlruns").as_uri())
    mlflow.set_experiment(exp)
    return mlflow


def _register(mlflow, cfg: dict, best_row: pd.Series, dataset: dict) -> dict:
    m = cfg["mlflow"]
    name, serving = m["model_name"], m.get("serving_alias", "champion")
    mv = mlflow.register_model(best_row["model_uri"], name)
    client = mlflow.MlflowClient()
    tags = {"candidate": best_row.name, **{f"data_{k}": str(v) for k, v in dataset.items()},
            **{k: str(v) for k, v in best_row.items() if k.endswith(("_mean", "_std"))}}
    for k, v in tags.items():
        client.set_model_version_tag(name, mv.version, k, v)
    client.set_registered_model_alias(name, "challenger", mv.version)
    promoted, reason = False, None
    metric = cfg.get("training", {}).get("selection_metric", "balanced_accuracy")
    score = float(best_row.get(f"{metric}_mean", 0) or 0)
    min_score = float(cfg.get("training", {}).get("min_score_to_serve", 0.65))
    try:
        client.get_model_version_by_alias(name, serving)
        reason = "un champion existe déjà"
    except Exception:  # pas encore de champion : le premier modèle le devient… s'il est assez bon
        if score >= min_score:
            client.set_registered_model_alias(name, serving, mv.version)
            promoted = True
        else:
            reason = f"{metric}={score:.2f} < {min_score} (min_score_to_serve) : l'app garde les règles du coach"
    return {"name": name, "version": mv.version, "promoted": promoted, "reason": reason}


def demote(cfg: dict, alias: str | None = None) -> None:
    """Retire l'alias servi : l'app repasse sur les règles du coach."""
    mlflow = _mlflow(cfg)
    alias = alias or cfg["mlflow"].get("serving_alias", "champion")
    mlflow.MlflowClient().delete_registered_model_alias(cfg["mlflow"]["model_name"], alias)


def list_versions(cfg: dict) -> list[dict]:
    mlflow = _mlflow(cfg)
    if not mlflow:
        return []
    client = mlflow.MlflowClient()
    name = cfg["mlflow"]["model_name"]
    try:
        versions = client.search_model_versions(f"name='{name}'")
    except Exception:
        return []
    out = []
    for v in sorted(versions, key=lambda v: int(v.version)):
        v = client.get_model_version(name, v.version)  # la recherche ne renvoie pas les alias
        out.append({"version": int(v.version), "aliases": list(v.aliases or []), **(v.tags or {})})
    return out


def promote(cfg: dict, version: int, alias: str | None = None) -> None:
    mlflow = _mlflow(cfg)
    alias = alias or cfg["mlflow"].get("serving_alias", "champion")
    mlflow.MlflowClient().set_registered_model_alias(cfg["mlflow"]["model_name"], alias, str(version))


# --------------------------------------------------------------------------
# Chargement du modèle servi
# --------------------------------------------------------------------------
def load_model(cfg: dict, ref: str | int | None = None) -> dict | None:
    """ref : alias ('champion', 'challenger'), numéro de version, ou 'local'.
    Renvoie {'model', 'label', 'meta'} ou None s'il n'y a aucun modèle."""
    ref = ref if ref is not None else cfg.get("mlflow", {}).get("serving_alias", "champion")
    if str(ref) == "none":  # règles du coach uniquement
        return None
    if str(ref) != "local":
        mlflow = _mlflow(cfg) if _mlflow_available() else None
        if mlflow:
            name = cfg["mlflow"]["model_name"]
            client = mlflow.MlflowClient()
            try:
                mv = (client.get_model_version(name, str(ref)) if str(ref).isdigit()
                      else client.get_model_version_by_alias(name, str(ref)))
                model = mlflow.sklearn.load_model(f"models:/{name}/{mv.version}")
                return {"model": model, "meta": {"version": int(mv.version), "aliases": list(mv.aliases or []),
                                                 **(mv.tags or {})},
                        "label": f"v{mv.version} · {mv.tags.get('candidate', '?')}"
                                 + (f" · {', '.join(mv.aliases)}" if mv.aliases else "")}
            except Exception:
                return None  # pas de champion : règles du coach
    p = Path(cfg["paths"]["models_dir"]) / "manchette_clf.joblib"
    if p.exists() and (str(ref) == "local" or not _mlflow_available()):
        import joblib
        b = joblib.load(p)
        return {"model": b["model"], "meta": b["meta"], "label": f"local · {b['meta'].get('candidate', '?')}"}
    return None


def _mlflow_available() -> bool:
    try:
        import mlflow  # noqa: F401
        return True
    except ImportError:
        return False


def predict_proba(bundle: dict, features: dict) -> float:
    model = bundle["model"]
    cols = list(getattr(model, "feature_names_in_", FEATURE_COLUMNS))
    x = pd.DataFrame([[features.get(c, np.nan) for c in cols]], columns=cols, dtype=float)
    return float(model.predict_proba(x)[0, 1])


def describe(board: pd.DataFrame) -> str:
    cols = ["candidate"] + [c for c in board.columns if c.endswith(("_mean", "_std"))]
    return board[cols].to_string(index=False)


def save_json(obj, path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
