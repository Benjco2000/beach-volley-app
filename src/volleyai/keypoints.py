"""Nettoyage des keypoints : interpolation, lissage, orientation profil."""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from volleyai.pose import KP

LEFT = [KP[n] for n in ("left_shoulder", "left_elbow", "left_wrist", "left_hip", "left_knee", "left_ankle")]
RIGHT = [KP[n] for n in ("right_shoulder", "right_elbow", "right_wrist", "right_hip", "right_knee", "right_ankle")]


def _interp_1d(v: np.ndarray) -> np.ndarray:
    ok = ~np.isnan(v)
    if ok.sum() == 0:
        return v
    if ok.sum() == 1:
        return np.full_like(v, v[ok][0])
    idx = np.arange(len(v))
    return np.interp(idx, idx[ok], v[ok])


def interpolate(series: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Interpole linéairement les trous (NaN) d'un tableau (T, ...) le long du temps."""
    out = series.astype(float).copy()
    if valid is not None:
        out[~valid] = np.nan
    flat = out.reshape(len(out), -1)
    for j in range(flat.shape[1]):
        flat[:, j] = _interp_1d(flat[:, j])
    return flat.reshape(out.shape)


def smooth(series: np.ndarray, fps: float, window_s: float) -> np.ndarray:
    win = int(round(window_s * fps)) | 1  # impair
    win = max(5, win)
    if len(series) <= win or np.isnan(series).all():
        return series
    out = series.copy()
    flat = out.reshape(len(out), -1)
    for j in range(flat.shape[1]):
        col = flat[:, j]
        if not np.isnan(col).any():
            flat[:, j] = savgol_filter(col, win, polyorder=2)
    return flat.reshape(series.shape)


def clean_keypoints(kps: np.ndarray, fps: float, conf_min: float = 0.3, window_s: float = 0.12) -> np.ndarray:
    """(T,17,3) -> (T,17,2) positions interpolées + lissées."""
    valid = kps[:, :, 2] >= conf_min
    xy = interpolate(kps[:, :, :2], np.repeat(valid[:, :, None], 2, axis=2))
    return smooth(xy, fps, window_s)


def near_side(kps: np.ndarray) -> str:
    """Côté le plus visible par la caméra (vue de profil)."""
    return "left" if np.nanmean(kps[:, LEFT, 2]) >= np.nanmean(kps[:, RIGHT, 2]) else "right"


def facing_sign(xy: np.ndarray) -> int:
    """+1 si le joueur regarde vers +x (droite de l'image), -1 sinon.
    En manchette, les poignets sont devant les hanches : signal robuste."""
    hip = xy[:, [KP["left_hip"], KP["right_hip"]], 0].mean(axis=1)
    wrist = xy[:, [KP["left_wrist"], KP["right_wrist"]], 0].mean(axis=1)
    nose = xy[:, KP["nose"], 0]
    def sgn(v: np.ndarray) -> float:
        m = np.nanmedian(v) if np.isfinite(v).any() else 0.0
        return float(np.sign(m))

    score = sgn(wrist - hip) * 2 + sgn(nose - hip)
    return 1 if score >= 0 else -1


def mirror_x(xy: np.ndarray, width: int) -> np.ndarray:
    out = xy.copy()
    out[..., 0] = width - out[..., 0]
    return out


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle ABC en degrés, vectorisé sur le temps. a, b, c : (T, 2)."""
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))
