"""Estimation de pose + détection du ballon (couche Silver).

Backend par défaut : YOLO11-pose (Ultralytics), 17 points COCO.
Sortie : keypoints (T, 17, 3) en pixels de la vidéo d'origine (x, y, conf),
ballon (T, 3) (x, y, conf) avec NaN quand non détecté.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from volleyai.video import ensure_readable, iter_frames, probe

COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
KP = {n: i for i, n in enumerate(COCO_KEYPOINTS)}
SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6),
]
SPORTS_BALL = 32


@dataclass
class PoseResult:
    keypoints: np.ndarray      # (T, 17, 3)
    ball: np.ndarray           # (T, 3)
    frame_indices: np.ndarray  # (T,) index dans la vidéo d'origine
    fps: float                 # fps effectif (après stride)
    width: int
    height: int
    source: str

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, keypoints=self.keypoints, ball=self.ball,
                            frame_indices=self.frame_indices,
                            meta=np.array([self.fps, self.width, self.height]),
                            source=np.array(self.source))

    @classmethod
    def load(cls, path: str | Path) -> "PoseResult":
        d = np.load(path, allow_pickle=False)
        fps, w, h = d["meta"]
        return cls(d["keypoints"], d["ball"], d["frame_indices"], float(fps), int(w), int(h), str(d["source"]))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class PoseExtractor:
    def __init__(self, pose_model: str = "yolo11n-pose.pt", ball_model: str | None = "yolo11n.pt",
                 detect_ball: bool = True, imgsz: int = 640, conf: float = 0.25,
                 ball_conf: float = 0.15, device: str | None = None, batch: int = 16):
        from ultralytics import YOLO

        self.pose = YOLO(pose_model)
        self.ball = YOLO(ball_model) if detect_ball and ball_model else None
        self.imgsz, self.conf, self.ball_conf = imgsz, conf, ball_conf
        self.device, self.batch = device, batch

    @classmethod
    def from_config(cls, cfg: dict) -> "PoseExtractor":
        p = cfg["pose"]
        return cls(p["pose_model"], p.get("ball_model"), p.get("detect_ball", True),
                   p.get("imgsz", 640), p.get("conf", 0.25), p.get("ball_conf", 0.15))

    # -- sélection du joueur principal ------------------------------------
    @staticmethod
    def _pick_person(boxes: np.ndarray, kps: np.ndarray, prev_box: np.ndarray | None) -> int | None:
        if len(boxes) == 0:
            return None
        if prev_box is not None:
            ious = np.array([_iou(b, prev_box) for b in boxes])
            if ious.max() > 0.2:
                return int(ious.argmax())
        # sinon : la plus grande personne, pondérée par la confiance des points
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return int((areas * kps[:, :, 2].mean(axis=1)).argmax())

    @staticmethod
    def _pick_ball(dets: np.ndarray, prev: np.ndarray | None) -> np.ndarray:
        """dets : (n, 3) centres x, y, conf."""
        if len(dets) == 0:
            return np.array([np.nan, np.nan, 0.0])
        if prev is not None and not np.isnan(prev[0]):
            d = np.hypot(dets[:, 0] - prev[0], dets[:, 1] - prev[1])
            return dets[int(d.argmin())]
        return dets[int(dets[:, 2].argmax())]

    def process(self, video_path: str | Path, stride: int = 1, max_side: int | None = 1280) -> PoseResult:
        src = ensure_readable(video_path)
        info = probe(src)
        kps_all, ball_all, idx_all = [], [], []
        prev_box, prev_ball = None, None

        frames, idxs, scales = [], [], []

        def flush():
            nonlocal prev_box, prev_ball
            if not frames:
                return
            pres = self.pose(frames, imgsz=self.imgsz, conf=self.conf, verbose=False, device=self.device)
            bres = (self.ball(frames, imgsz=self.imgsz, conf=self.ball_conf, classes=[SPORTS_BALL],
                              verbose=False, device=self.device) if self.ball else [None] * len(frames))
            for r, br, i, s in zip(pres, bres, idxs, scales):
                boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
                kps = r.keypoints.data.cpu().numpy() if r.keypoints is not None else np.zeros((0, 17, 3))
                j = self._pick_person(boxes, kps, prev_box)
                if j is None:
                    kp = np.full((17, 3), np.nan)
                    kp[:, 2] = 0.0
                else:
                    kp = kps[j].copy()
                    kp[:, :2] /= s
                    prev_box = boxes[j]
                if br is not None and br.boxes is not None and len(br.boxes):
                    bb = br.boxes.xyxy.cpu().numpy() / s
                    dets = np.c_[(bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2, br.boxes.conf.cpu().numpy()]
                else:
                    dets = np.zeros((0, 3))
                ball = self._pick_ball(dets, prev_ball)
                prev_ball = ball if not np.isnan(ball[0]) else prev_ball
                kps_all.append(kp)
                ball_all.append(ball)
                idx_all.append(i)
            frames.clear(); idxs.clear(); scales.clear()

        for i, frame in iter_frames(src, stride=stride):
            h, w = frame.shape[:2]
            s = min(1.0, max_side / max(h, w)) if max_side else 1.0
            if s < 1:
                import cv2
                frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            frames.append(frame); idxs.append(i); scales.append(s)
            if len(frames) >= self.batch:
                flush()
        flush()

        return PoseResult(np.asarray(kps_all, dtype=float).reshape(-1, 17, 3),
                          np.asarray(ball_all, dtype=float).reshape(-1, 3),
                          np.asarray(idx_all), info.fps / stride, info.width, info.height, str(video_path))
