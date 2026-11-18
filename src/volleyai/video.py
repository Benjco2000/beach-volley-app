"""Lecture / écriture vidéo (iPhone .MOV HEVC compris)."""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    fps: float
    n_frames: int
    width: int
    height: int


def _ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _readable(path: str) -> bool:
    cap = cv2.VideoCapture(path)
    ok, _ = cap.read()
    cap.release()
    return ok


def ensure_readable(path: str | Path) -> str:
    """Si OpenCV ne sait pas décoder le fichier (HEVC selon les builds),
    on le transcode en H.264 dans un fichier temporaire."""
    path = str(path)
    if _readable(path):
        return path
    out = Path(tempfile.mkdtemp()) / (Path(path).stem + "_h264.mp4")
    subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-i", path, "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "20", "-an", str(out)], check=True)
    return str(out)


def probe(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame = cap.read()  # dimensions après rotation automatique iPhone
    cap.release()
    h, w = frame.shape[:2] if ok else (0, 0)
    return VideoInfo(str(path), float(fps), n, w, h)


def iter_frames(path: str | Path, stride: int = 1, max_side: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Itère (index_frame, image BGR). Redimensionne si max_side est donné."""
    cap = cv2.VideoCapture(str(path))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            if max_side:
                h, w = frame.shape[:2]
                s = max_side / max(h, w)
                if s < 1:
                    frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            yield i, frame
        i += 1
    cap.release()


class H264Writer:
    """Écrit un MP4 H.264 lisible par les navigateurs (Streamlit)."""

    def __init__(self, path: str | Path, width: int, height: int, fps: float):
        import imageio_ffmpeg

        self.size = (width - width % 2, height - height % 2)
        self._gen = imageio_ffmpeg.write_frames(
            str(path), self.size, fps=fps, codec="libx264", pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p", macro_block_size=1, quality=7,
            ffmpeg_log_level="error",
        )
        self._gen.send(None)

    def write(self, frame_bgr: np.ndarray) -> None:
        frame = frame_bgr[: self.size[1], : self.size[0]]
        self._gen.send(np.ascontiguousarray(frame[:, :, ::-1]))

    def close(self) -> None:
        self._gen.close()
