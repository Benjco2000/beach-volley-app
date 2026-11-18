"""Critères du coach, évalués sur les features (explicables, sans données)."""
from __future__ import annotations

import math
import operator
from dataclasses import asdict, dataclass

OPS = {"<=": operator.le, ">=": operator.ge, "<": operator.lt, ">": operator.gt}


@dataclass
class CriterionResult:
    id: str
    label: str
    status: str            # "ok" | "ko" | "na" (non évaluable)
    value: float | None
    threshold: float
    op: str
    weight: float
    message: str | None    # explication si ko / na
    axis: str | None       # axe d'amélioration si ko
    margin: float | None   # écart au seuil, normalisé (>0 = du mauvais côté)

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate(features: dict, criteria: list[dict]) -> list[CriterionResult]:
    results = []
    for c in criteria:
        v = features.get(c["feature"])
        valid = v is not None and not (isinstance(v, float) and math.isnan(v))
        if not valid:
            reason = ("Ballon non détecté : critère non évaluable." if c.get("requires_ball")
                      else "Mesure indisponible (points du corps mal détectés).")
            results.append(CriterionResult(c["id"], c["label"], "na", None, c["threshold"], c["op"],
                                           c.get("weight", 1), reason, None, None))
            continue
        ok = OPS[c["op"]](v, c["threshold"])
        scale = abs(c["threshold"]) or 1.0
        margin = (v - c["threshold"]) / scale * (1 if c["op"] in ("<=", "<") else -1)
        results.append(CriterionResult(
            c["id"], c["label"], "ok" if ok else "ko", round(float(v), 3), c["threshold"], c["op"],
            c.get("weight", 1), None if ok else c["fail_message"], None if ok else c["axis"],
            round(float(margin), 3),
        ))
    return results


def rule_score(results: list[CriterionResult]) -> float | None:
    """Part pondérée des critères évaluables qui sont respectés."""
    evaluable = [r for r in results if r.status != "na"]
    if not evaluable:
        return None
    total = sum(r.weight for r in evaluable)
    return sum(r.weight for r in evaluable if r.status == "ok") / total


def improvement_axes(results: list[CriterionResult], max_axes: int = 3) -> list[dict]:
    """Critères ratés, triés par importance (poids × écart au seuil)."""
    ko = [r for r in results if r.status == "ko"]
    ko.sort(key=lambda r: r.weight * (1 + max(0.0, r.margin or 0)), reverse=True)
    return [{"critere": r.label, "constat": r.message, "axe": r.axis} for r in ko[:max_axes]]
