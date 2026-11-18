"""Chargement de la configuration YAML."""
from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "manchette.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # chemins relatifs -> relatifs à la racine du repo
    for key, value in cfg.get("paths", {}).items():
        p = Path(value)
        cfg["paths"][key] = p if p.is_absolute() else ROOT / p
    sa = cfg.get("drive", {}).get("service_account_file")
    if sa and not Path(sa).is_absolute():
        cfg["drive"]["service_account_file"] = ROOT / sa
    return cfg


def normalize_name(name: str) -> str:
    """'Réussie ' -> 'reussie' (minuscules, sans accents ni espaces superflus)."""
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).strip().lower()
