"""Orchestration : vidéo -> pose -> features -> règles + modèle -> rapport."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pandas as pd

from volleyai.analysis.feedback import llm_feedback, text_feedback
from volleyai.analysis.reference import compare_to_reference, load_reference
from volleyai.analysis.rules import evaluate, improvement_axes, rule_score
from volleyai.features.manchette import FEATURE_COLUMNS, analyze_pose
from volleyai.model.train import load_model, predict_proba
from volleyai.pose import PoseExtractor, PoseResult

REFERENCE_FILE = "reference_manchette.json"


def _file_key(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.md5()
    with open(p, "rb") as f:
        h.update(f.read(1 << 20))
    h.update(str(p.stat().st_size).encode())
    return h.hexdigest()[:16]


def _clean(v):
    return None if isinstance(v, float) and math.isnan(v) else v


class Analyzer:
    def __init__(self, cfg: dict, extractor: PoseExtractor | None = None):
        self.cfg = cfg
        self._extractor = extractor
        self.model = load_model(cfg["paths"]["models_dir"])
        self.reference = load_reference(Path(cfg["paths"]["models_dir"]) / REFERENCE_FILE)

    @property
    def extractor(self) -> PoseExtractor:
        if self._extractor is None:
            self._extractor = PoseExtractor.from_config(self.cfg)
        return self._extractor

    # -- étape 1 : pose (avec cache .npz) ----------------------------------
    def pose(self, video_path: str | Path, cache: bool = True) -> PoseResult:
        cache_path = Path(self.cfg["paths"]["interim_dir"]) / f"{_file_key(video_path)}.npz"
        if cache and cache_path.exists():
            return PoseResult.load(cache_path)
        p = self.cfg["pose"]
        res = self.extractor.process(video_path, stride=p.get("frame_stride", 1), max_side=p.get("max_side", 1280))
        if cache:
            res.save(cache_path)
        return res

    # -- étape 2 : diagnostic d'une répétition -----------------------------
    def diagnose(self, feats: dict, use_llm: bool | None = None) -> dict:
        criteria = evaluate(feats, self.cfg["criteria"])
        score = rule_score(criteria)
        vcfg = self.cfg["verdict"]
        proba = predict_proba(self.model, feats) if self.model else None
        total_w = sum(c.weight for c in criteria) or 1
        coverage = sum(c.weight for c in criteria if c.status != "na") / total_w
        if coverage < vcfg.get("min_coverage", 0.6):
            verdict, confidence, source = "indetermine", 0.0, "couverture_insuffisante"
        elif proba is not None:
            verdict = "reussie" if proba >= vcfg["model_threshold"] else "ratee"
            confidence, source = max(proba, 1 - proba), "modele"
        elif score is not None:
            verdict = "reussie" if score >= vcfg["rule_score_threshold"] else "ratee"
            confidence, source = score if verdict == "reussie" else 1 - score, "regles"
        else:
            verdict, confidence, source = "indetermine", 0.0, "aucune"

        warnings = []
        if verdict == "indetermine":
            warnings.append(f"Seulement {coverage:.0%} des critères sont mesurables (corps coupé, "
                            "jambes hors champ ?) : pas de verdict.")
        if feats.get("pose_quality", 1) < 0.4:
            warnings.append("Squelette mal détecté (joueur trop loin, coupé ou de dos) : résultat peu fiable.")
        if feats.get("contact_method") == "fallback":
            warnings.append("Instant de contact non détecté avec certitude.")
        if feats.get("contact_method") == "pose":
            warnings.append("Ballon non détecté : instant de contact estimé à partir de la posture.")

        report = {
            "rep": feats["rep"],
            "contact_time_s": feats["contact_time_s"],
            "contact_frame": feats["contact_frame"],
            "contact_method": feats["contact_method"],
            "verdict": verdict, "confidence": round(float(confidence), 3), "source": source,
            "rule_score": None if score is None else round(score, 3),
            "coverage": round(coverage, 3),
            "model_proba": None if proba is None else round(proba, 3),
            "criteria": [c.to_dict() for c in criteria],
            "axes": improvement_axes(criteria),
            "deviations": compare_to_reference(feats, self.reference) if self.reference else [],
            "features": {k: _clean(feats.get(k)) for k in FEATURE_COLUMNS},
            "warnings": warnings,
        }
        report["feedback"] = text_feedback(report)
        llm_on = self.cfg.get("llm", {}).get("enabled") if use_llm is None else use_llm
        if llm_on:
            txt = llm_feedback(report, self.cfg["llm"]["model"], self.cfg["llm"].get("max_tokens", 600))
            if txt:
                report["feedback_llm"] = txt
        return report

    # -- bout en bout -------------------------------------------------------
    def analyze(self, video_path: str | Path, overlay_path: str | Path | None = None,
                use_llm: bool | None = None) -> dict:
        pose = self.pose(video_path)
        series, reps = analyze_pose(pose, self.cfg)
        reports = [self.diagnose(f, use_llm) for f in reps]
        out = {
            "video": str(video_path), "fps": pose.fps, "n_frames": int(len(pose.keypoints)),
            "camera_side": series.side, "n_reps": len(reports), "reps": reports,
            "summary": {
                "reussies": sum(r["verdict"] == "reussie" for r in reports),
                "indeterminees": sum(r["verdict"] == "indetermine" for r in reports),
                "ratees": sum(r["verdict"] == "ratee" for r in reports),
            },
        }
        if overlay_path:
            from volleyai.overlay import render
            out["snapshots"] = render(video_path, pose, reports, overlay_path)
            out["overlay"] = str(overlay_path)
        out["_series"] = series  # pour les graphiques de l'app (non sérialisé)
        return out


def main_rep(reps: list[dict]) -> dict | None:
    """Répétition principale d'un clip d'entraînement (1 manchette par clip) :
    on privilégie un contact détecté avec le ballon, puis la meilleure qualité."""
    if not reps:
        return None
    rank = {"ball": 2, "pose": 1, "fallback": 0}
    return max(reps, key=lambda r: (rank.get(r["contact_method"], 0), r.get("pose_quality", 0)))


def build_feature_table(manifest: pd.DataFrame, analyzer: Analyzer, out_csv: str | Path,
                        all_reps: bool = False) -> pd.DataFrame:
    rows = []
    for _, m in manifest.iterrows():
        try:
            pose = analyzer.pose(m["local_path"])
            _, reps = analyze_pose(pose, analyzer.cfg)
        except Exception as e:  # une vidéo illisible ne bloque pas le lot
            print(f"[extract] ERREUR {m['name']}: {e}")
            continue
        chosen = reps if all_reps else [r for r in [main_rep(reps)] if r]
        for r in chosen:
            rows.append({**m.to_dict(), **r, "n_reps_detected": len(reps)})
        print(f"[extract] {m.get('folder_path', '')}/{m['name']}: {len(reps)} contact(s)")
    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df
