"""Vidéo annotée : squelette, ballon, instants de contact et verdict."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from volleyai.pose import SKELETON, PoseResult
from volleyai.video import H264Writer, ensure_readable, iter_frames

GREEN, RED, YELLOW, WHITE = (80, 200, 80), (60, 60, 230), (0, 215, 255), (255, 255, 255)


def _draw_pose(img: np.ndarray, kp: np.ndarray, color, conf_min: float = 0.3) -> None:
    t = max(2, img.shape[0] // 300)
    for a, b in SKELETON:
        if kp[a, 2] >= conf_min and kp[b, 2] >= conf_min:
            cv2.line(img, tuple(int(v) for v in kp[a, :2]), tuple(int(v) for v in kp[b, :2]), color, t, cv2.LINE_AA)
    for x, y, c in kp:
        if c >= conf_min:
            cv2.circle(img, (int(x), int(y)), t + 2, WHITE, -1, cv2.LINE_AA)


def _label(img: np.ndarray, text: str, org: tuple[int, int], color=WHITE) -> None:
    s = img.shape[0] / 900
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), int(4 * s) + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, s, color, int(2 * s) + 1, cv2.LINE_AA)


def render(video_path: str | Path, pose: PoseResult, reps: list[dict], out_path: str | Path,
           max_side: int = 960) -> dict[int, np.ndarray]:
    """Écrit la vidéo annotée ; renvoie {rep: image du contact} pour l'affichage."""
    src = ensure_readable(video_path)
    contact_frames = {r["contact_frame"]: r for r in reps}
    window = max(2, int(pose.fps * 0.05))
    snapshots: dict[int, np.ndarray] = {}
    writer = None
    pi = 0
    for i, frame in iter_frames(src):
        h, w = frame.shape[:2]
        s = min(1.0, max_side / max(h, w))
        if s < 1:
            frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        if writer is None:
            writer = H264Writer(out_path, frame.shape[1], frame.shape[0], pose.fps * max(1, _stride(pose)))
        while pi + 1 < len(pose.frame_indices) and pose.frame_indices[pi + 1] <= i:
            pi += 1
        # verdict de la répétition la plus proche
        rep = min(reps, key=lambda r: abs(r["contact_frame"] - i)) if reps else None
        color = {"reussie": GREEN, "ratee": RED}.get(rep.get("verdict") if rep else None, YELLOW)
        kp = pose.keypoints[pi].copy()
        kp[:, :2] *= s
        _draw_pose(frame, kp, color)
        bx, by, bc = pose.ball[pi]
        if not np.isnan(bx):
            cv2.circle(frame, (int(bx * s), int(by * s)), max(6, frame.shape[0] // 60), YELLOW, 2, cv2.LINE_AA)
        near = [f for f in contact_frames if abs(f - i) <= window * max(1, _stride(pose))]
        if near:
            r = contact_frames[near[0]]
            _label(frame, f"CONTACT #{r['rep'] + 1}", (20, frame.shape[0] - 30), YELLOW)
            if near[0] == i or r["rep"] not in snapshots:
                snapshots[r["rep"]] = frame.copy()
        if rep:
            txt = f"#{rep['rep'] + 1} " + {"reussie": "REUSSIE", "ratee": "A CORRIGER"}.get(rep.get("verdict"), "INDETERMINEE")
            _label(frame, txt, (20, 40), color)
        writer.write(frame)
    if writer:
        writer.close()
    return snapshots


def _stride(pose: PoseResult) -> int:
    if len(pose.frame_indices) < 2:
        return 1
    return int(max(1, np.median(np.diff(pose.frame_indices))))
