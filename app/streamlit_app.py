"""App : uploader une vidéo de manchette -> verdict + axes d'amélioration.

Lancer : streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from volleyai.config import load_config
from volleyai.pipeline import Analyzer

st.set_page_config(page_title="Analyse manchette", page_icon="🏐", layout="wide")

STATUS = {"ok": "✅", "ko": "❌", "na": "➖"}


@st.cache_resource
def get_analyzer(model_ref: str) -> Analyzer:
    return Analyzer(load_config(), model_ref=model_ref)


@st.cache_data(ttl=30)
def model_options() -> dict[str, str]:
    """{libellé: référence} : versions du registre MLflow + règles seules."""
    from volleyai.model.train import list_versions
    opts = {}
    versions = list_versions(load_config())
    for v in sorted(versions, key=lambda v: ("champion" not in v["aliases"], -v["version"])):
        ba = v.get("balanced_accuracy_mean")
        tag = f" [{', '.join(v['aliases'])}]" if v["aliases"] else ""
        src = " (synthétique)" if v.get("data_source") == "synthetic" else ""
        opts[f"v{v['version']} · {v.get('candidate', '?')}{tag} · BA {ba}{src}"] = str(v["version"])
    opts["Règles du coach uniquement"] = "none"
    return opts


st.title("🏐 Analyse de manchette")
st.caption("Vidéo de profil, un joueur bien visible, ballon dans le champ si possible.")

with st.sidebar:
    st.subheader("Modèle utilisé")
    opts = model_options()
    choice = st.selectbox("Version", list(opts), index=0,
                          help="Registre MLflow. Le champion est proposé par défaut.")
    an = get_analyzer(opts[choice])
    if an.model:
        m = an.model["meta"]
        st.caption(f"Précision équilibrée (CV) : {m.get('balanced_accuracy_mean', 'n/a')} "
                   f"± {m.get('balanced_accuracy_std', 'n/a')} — {m.get('data_n_samples', '?')} manchettes")
        if m.get("data_source") == "synthetic":
            st.warning("Modèle entraîné sur des données synthétiques : démonstration uniquement.")
    else:
        st.info("Verdict basé sur les critères du coach (seuils de configs/manchette.yaml).")
    use_llm = st.toggle("Reformuler avec Claude (ANTHROPIC_API_KEY)", value=False)

up = st.file_uploader("Vidéo (.mov, .mp4)", type=["mov", "mp4", "m4v", "avi"])
if not up:
    st.stop()

tmp = Path(tempfile.mkdtemp())
video = tmp / up.name
video.write_bytes(up.getbuffer())
overlay = tmp / "annotee.mp4"

with st.spinner("Détection du squelette et du ballon, calcul des mesures…"):
    report = an.analyze(video, overlay, use_llm=use_llm)

if not report["reps"]:
    st.error("Aucune manchette détectée dans la vidéo.")
    st.stop()

c1, c2 = st.columns([3, 2])
with c1:
    st.video(str(overlay))
with c2:
    s = report["summary"]
    st.metric("Manchettes détectées", report["n_reps"],
              f"{s['reussies']} réussie(s) / {s['ratees']} à corriger", delta_color="off")

series = report["_series"]
for r in report["reps"]:
    label = {"reussie": "✅ réussie", "ratee": "❌ à corriger"}.get(r["verdict"], "❔ indéterminée")
    title = f"Manchette #{r['rep'] + 1} — {label} " \
            f"({r['confidence']:.0%}, via {r['source']}) — contact à {r['contact_time_s']} s"
    with st.expander(title, expanded=True):
        for w in r["warnings"]:
            st.warning(w)
        left, right = st.columns([2, 3])
        with left:
            snap = report.get("snapshots", {}).get(r["rep"])
            if snap is not None:
                st.image(cv2.cvtColor(snap, cv2.COLOR_BGR2RGB), caption="Instant du contact")
        with right:
            st.markdown(r.get("feedback_llm") or r["feedback"].replace("\n", "  \n"))
            crit = pd.DataFrame([{
                "": STATUS[c["status"]], "Critère": c["label"],
                "Mesure": c["value"], "Seuil": f"{c['op']} {c['threshold']}",
            } for c in r["criteria"]])
            st.dataframe(crit, hide_index=True, width="stretch")
            if r["deviations"]:
                st.caption("Écarts au profil des manchettes réussies")
                st.dataframe(pd.DataFrame(r["deviations"])[["description", "valeur", "plage_reussies"]],
                             hide_index=True, width="stretch")

        # Courbes autour du contact : bras vs hanches (swing, synchro, redressement)
        c, fps = r["contact_frame"], series.fps
        idx = int(np.searchsorted(series.frame_indices, c))
        a, b = max(0, idx - int(fps)), min(len(series.knee), idx + int(0.6 * fps))
        t = (np.arange(a, b) - idx) / fps
        chart = pd.DataFrame({
            "t (s)": t,
            "Angle bras/tronc (°)": series.arm_trunk[a:b],
            "Angle genou (°)": series.knee[a:b],
            "Hauteur hanches (x100 tronc)": series.up["hip"][a:b] * 100,
        }).set_index("t (s)")
        st.line_chart(chart, height=220)

with st.expander("Détails techniques (JSON)"):
    st.json({k: v for k, v in report.items() if k not in ("_series", "snapshots")}, expanded=False)
