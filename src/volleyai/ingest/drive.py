"""Ingestion Google Drive (couche Bronze).

Parcourt récursivement un dossier Drive, télécharge les vidéos et écrit un
manifeste CSV. Idempotent : une vidéo déjà téléchargée avec le même md5 n'est
pas re-téléchargée.

Authentification : compte de service Google Cloud. Partage le dossier
"BEACH VOLLEY" (lecture seule) avec l'adresse e-mail du compte de service.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from volleyai.config import normalize_name

FOLDER_MIME = "application/vnd.google-apps.folder"
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


@dataclass
class DriveVideo:
    file_id: str
    name: str
    md5: str | None
    size: int | None
    folder_path: str          # ex. "réussie" ou "Csilla/06/10 - Manchettes"
    label: int | None         # 1 réussie, 0 ratée, None inconnu (test)
    player: str | None        # pour G3 : nom du dossier joueuse
    local_path: str | None = None


def build_service(service_account_file: str | Path):
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    creds = service_account.Credentials.from_service_account_file(
        str(service_account_file), scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _list_children(service, folder_id: str) -> list[dict]:
    items, token = [], None
    while True:
        resp = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields="nextPageToken, files(id, name, mimeType, md5Checksum, size)",
                pageSize=1000,
                pageToken=token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        items.extend(resp.get("files", []))
        token = resp.get("nextPageToken")
        if not token:
            return items


def walk_videos(service, folder_id: str, path: tuple[str, ...] = ()) -> list[tuple[dict, tuple[str, ...]]]:
    out = []
    for item in _list_children(service, folder_id):
        if item["mimeType"] == FOLDER_MIME:
            out.extend(walk_videos(service, item["id"], path + (item["name"],)))
        elif item["mimeType"].startswith("video/"):
            out.append((item, path))
    return out


def list_training_videos(service, folder_id: str, label_map: dict[str, int]) -> list[DriveVideo]:
    """Dossier d'entraînement : le premier sous-dossier donne le label."""
    videos = []
    for item, path in walk_videos(service, folder_id):
        label = label_map.get(normalize_name(path[0])) if path else None
        if label is None:
            continue  # vidéo hors des sous-dossiers labellisés
        videos.append(DriveVideo(item["id"], item["name"], item.get("md5Checksum"),
                                 int(item.get("size", 0)), "/".join(path), label, None))
    return videos


def list_test_videos(service, folder_id: str, subfolder_filter: str | None) -> list[DriveVideo]:
    """Dossier G3 : <Joueuse>/<date - geste>/clip."""
    videos = []
    for item, path in walk_videos(service, folder_id):
        if subfolder_filter and not any(subfolder_filter in normalize_name(p) for p in path):
            continue
        player = path[0] if path else None
        videos.append(DriveVideo(item["id"], item["name"], item.get("md5Checksum"),
                                 int(item.get("size", 0)), "/".join(path), None, player))
    return videos


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(service, video: DriveVideo, dest_dir: Path) -> Path:
    from googleapiclient.http import MediaIoBaseDownload

    dest_dir.mkdir(parents=True, exist_ok=True)
    # file_id en préfixe : plusieurs fichiers Drive ont le même nom (ex. IMG_6347.mov)
    dest = dest_dir / f"{video.file_id}_{video.name}"
    if dest.exists() and (video.md5 is None or _md5(dest) == video.md5):
        return dest
    request = service.files().get_media(fileId=video.file_id, supportsAllDrives=True)
    with io.FileIO(dest, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=8 << 20)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return dest


def ingest(cfg: dict, split: str = "train") -> pd.DataFrame:
    """Télécharge les vidéos d'un split ('train' ou 'test') et écrit le manifeste."""
    dcfg = cfg["drive"]
    service = build_service(dcfg["service_account_file"])
    if split == "train":
        videos = list_training_videos(service, dcfg["training_folder_id"], dcfg["label_map"])
    else:
        videos = list_test_videos(service, dcfg["test_folder_id"], dcfg.get("test_subfolder_filter"))

    dest_root = Path(cfg["paths"]["raw_dir"]) / split
    for v in videos:
        v.local_path = str(download(service, v, dest_root))
        print(f"[ingest] {split} {v.folder_path}/{v.name} -> ok")

    manifest = pd.DataFrame([asdict(v) for v in videos])
    out = Path(cfg["paths"]["raw_dir"]) / f"manifest_{split}.csv"
    manifest.to_csv(out, index=False)
    print(f"[ingest] {len(manifest)} vidéos, manifeste : {out}")
    return manifest


def manifest_from_local(root: str | Path, label_map: dict[str, int]) -> pd.DataFrame:
    """Alternative sans API : construit le manifeste depuis un dossier local
    organisé comme le Drive (ex. téléchargé à la main, ou monté via Drive Desktop)."""
    root = Path(root)
    rows = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in {".mov", ".mp4", ".m4v", ".avi"}:
            continue
        rel = p.relative_to(root).parts[:-1]
        label = label_map.get(normalize_name(rel[0])) if rel else None
        rows.append(dict(file_id=hashlib.md5(str(p).encode()).hexdigest()[:12], name=p.name,
                         md5=None, size=p.stat().st_size, folder_path="/".join(rel),
                         label=label, player=rel[0] if rel and label is None else None,
                         local_path=str(p)))
    return pd.DataFrame(rows)
