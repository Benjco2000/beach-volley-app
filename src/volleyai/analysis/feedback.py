"""Mise en mots du diagnostic.

Par défaut : texte déterministe construit à partir des critères ratés.
Option : un LLM Claude reformule le diagnostic structuré en conseil de coach.
Le LLM ne voit que des mesures (pas la vidéo) et ne décide pas du verdict :
il ne peut donc pas inventer un défaut qui n'a pas été mesuré.
"""
from __future__ import annotations

import json
import os

SYSTEM_PROMPT = """Tu es entraîneur de beach-volley. On te donne le diagnostic chiffré \
d'une manchette (critères respectés / ratés, mesures, écarts à la référence). \
Rédige un retour pour la joueuse : 1 phrase de verdict, puis 1 à 3 axes \
d'amélioration concrets, chacun avec un exercice ou une consigne simple. \
N'invente aucun défaut absent du diagnostic. Si des critères sont non \
évaluables, dis-le en une phrase. Tutoiement, ton bienveillant, 120 mots max."""


def text_feedback(report: dict) -> str:
    rep = report
    if rep["verdict"] == "indetermine":
        lines = ["Verdict : impossible de juger cette manchette (trop de mesures manquantes)."]
    else:
        lines = [f"Verdict : manchette {'réussie' if rep['verdict'] == 'reussie' else 'à corriger'} "
                 f"(confiance {rep['confidence']:.0%})."]
    if rep["axes"]:
        lines.append("Axes d'amélioration :")
        lines += [f"- {a['critere']} : {a['constat']} → {a['axe']}" for a in rep["axes"]]
    else:
        lines.append("Tous les critères évaluables sont respectés.")
    na = [c["label"] for c in rep["criteria"] if c["status"] == "na"]
    if na:
        lines.append(f"Non évaluable sur cette vidéo : {', '.join(na)}.")
    return "\n".join(lines)


def llm_feedback(report: dict, model: str = "claude-sonnet-5-5", max_tokens: int = 600) -> str | None:
    """Retourne None si le SDK ou la clé API ne sont pas disponibles."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except ImportError:
        return None
    payload = {k: report[k] for k in ("verdict", "confidence", "criteria", "axes", "deviations") if k in report}
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model, max_tokens=max_tokens, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=1)}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
