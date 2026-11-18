"""Ligne de commande.

  volleyai ingest  --split train|test        Drive -> data/raw + manifeste
  volleyai extract --split train|test        vidéos -> pose -> features (CSV)
  volleyai train                             features -> modèle + rapport de référence
  volleyai analyze VIDEO [--overlay out.mp4] diagnostic d'une vidéo
  volleyai evaluate-g3                       diagnostic de toutes les vidéos G3 -> CSV
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from volleyai.config import load_config


def _manifest(cfg: dict, split: str, local_dir: str | None) -> pd.DataFrame:
    if local_dir:
        from volleyai.config import normalize_name
        from volleyai.ingest.drive import manifest_from_local
        df = manifest_from_local(local_dir, cfg["drive"]["label_map"])
        filt = cfg["drive"].get("test_subfolder_filter")
        if split == "test" and filt and not df.empty:
            df = df[df["folder_path"].map(lambda p: filt in normalize_name(p))]
        return df
    path = Path(cfg["paths"]["raw_dir"]) / f"manifest_{split}.csv"
    if not path.exists():
        raise SystemExit(f"Manifeste absent ({path}). Lance d'abord `volleyai ingest --split {split}` "
                         f"ou passe --local-dir.")
    return pd.read_csv(path)


def cmd_ingest(cfg, args):
    from volleyai.ingest.drive import ingest
    ingest(cfg, args.split)


def cmd_extract(cfg, args):
    from volleyai.pipeline import Analyzer, build_feature_table
    manifest = _manifest(cfg, args.split, args.local_dir)
    out = Path(cfg["paths"]["processed_dir"]) / f"features_{args.split}.csv"
    df = build_feature_table(manifest, Analyzer(cfg), out, all_reps=args.split == "test")
    print(f"[extract] {len(df)} lignes -> {out}")


def cmd_train(cfg, args):
    from volleyai.analysis.reference import build_reference, save_reference
    from volleyai.model.train import train
    from volleyai.pipeline import REFERENCE_FILE
    features = Path(cfg["paths"]["processed_dir"]) / "features_train.csv"
    meta = train(features, cfg["paths"]["models_dir"], C=args.C)
    df = pd.read_csv(features)
    ref = build_reference(df[df["label"].isin([0, 1])])
    save_reference(ref, Path(cfg["paths"]["models_dir"]) / REFERENCE_FILE)
    save_reference(ref, Path(cfg["paths"]["reports_dir"]) / REFERENCE_FILE)
    print(json.dumps(meta["cv"], indent=2, ensure_ascii=False))
    print("\nFeatures les plus discriminantes (réussie vs ratée) :")
    for name, r in list(ref["features"].items())[:8]:
        print(f"  {name:28s} AUC={r['auc']}  réussies médiane={r['reussie_median']:.2f} "
              f"[{r['reussie_p10']:.2f}, {r['reussie_p90']:.2f}]  seuil suggéré={r['seuil_suggere']}")


def cmd_analyze(cfg, args):
    from volleyai.pipeline import Analyzer
    rep = Analyzer(cfg).analyze(args.video, args.overlay, use_llm=args.llm or None)
    rep.pop("_series", None); rep.pop("snapshots", None)
    for r in rep["reps"]:
        print(f"\n=== Manchette #{r['rep'] + 1} (t={r['contact_time_s']}s, contact via {r['contact_method']}) ===")
        print(r.get("feedback_llm") or r["feedback"])
        for w in r["warnings"]:
            print(f"⚠ {w}")
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def cmd_evaluate_g3(cfg, args):
    from volleyai.pipeline import Analyzer
    manifest = _manifest(cfg, "test", args.local_dir)
    an = Analyzer(cfg)
    rows = []
    for _, m in manifest.iterrows():
        try:
            rep = an.analyze(m["local_path"])
        except Exception as e:
            print(f"[g3] ERREUR {m['name']}: {e}")
            continue
        for r in rep["reps"]:
            rows.append({"player": m.get("player"), "video": m["name"], "rep": r["rep"] + 1,
                         "t_contact_s": r["contact_time_s"], "verdict": r["verdict"],
                         "confiance": r["confidence"], "source": r["source"],
                         "axes": " | ".join(a["critere"] for a in r["axes"]),
                         "alertes": " | ".join(r["warnings"])})
        print(f"[g3] {m.get('player')} {m['name']}: {rep['summary']}")
    out = Path(cfg["paths"]["reports_dir"]) / "g3_manchettes.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    if not df.empty:
        print(df.groupby("player")["verdict"].value_counts().unstack(fill_value=0))
    print(f"-> {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="volleyai")
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest"); p.add_argument("--split", choices=["train", "test"], default="train")
    p = sub.add_parser("extract"); p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--local-dir", help="dossier local organisé comme le Drive (sans API)")
    p = sub.add_parser("train"); p.add_argument("--C", type=float, default=0.5)
    p = sub.add_parser("analyze"); p.add_argument("video"); p.add_argument("--overlay")
    p.add_argument("--json"); p.add_argument("--llm", action="store_true")
    p = sub.add_parser("evaluate-g3"); p.add_argument("--local-dir")

    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    {"ingest": cmd_ingest, "extract": cmd_extract, "train": cmd_train,
     "analyze": cmd_analyze, "evaluate-g3": cmd_evaluate_g3}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
