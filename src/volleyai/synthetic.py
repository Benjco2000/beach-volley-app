"""Générateur de manchettes synthétiques (squelette 2D de profil).

Sert à tester la chaîne features -> règles -> modèle sans vidéo, avec des
défauts contrôlés (swing, saut, redressement, contact au coude...).
"""
from __future__ import annotations

import numpy as np

from volleyai.pose import KP, PoseResult

W, H = 1080, 1920
GROUND = 1700.0
TORSO, THIGH, SHANK, ARM = 260.0, 230.0, 230.0, 330.0


def _ramp(t, t0, t1, v0, v1):
    a = np.clip((t - t0) / max(t1 - t0, 1e-6), 0, 1)
    a = a * a * (3 - 2 * a)  # smoothstep
    return v0 + (v1 - v0) * a


def _rot(v: np.ndarray, deg: np.ndarray) -> np.ndarray:
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], axis=1)


def make_manchette(fps: float = 60, duration: float = 2.4, t_contact: float = 1.4,
                   swing: float = 8.0, jump: float = 0.0, rise: bool = False,
                   forearm_t: float = 0.3, arms_early: float = 0.0, split: float = 120.0,
                   bent_elbows: float = 0.0, facing: int = 1, with_ball: bool = True,
                   noise: float = 0.0, seed: int = 0) -> tuple[PoseResult, int]:
    """Renvoie (PoseResult, index du contact)."""
    rng = np.random.default_rng(seed)
    T = int(duration * fps)
    t = np.arange(T) / fps
    tc = t_contact

    # Genou : descente en préparation, poussée vers le contact, (option) redressement
    knee = _ramp(t, 0.2, tc - 0.45, 172, 118) + _ramp(t, tc - 0.25, tc + 0.05, 0, 22)
    if rise:
        knee = knee + _ramp(t, tc - 0.05, tc + 0.2, 0, 175 - 140)
    leg = (THIGH + SHANK) * np.sin(np.radians(knee) / 2)  # hauteur hanche au-dessus de la cheville
    x0 = 500.0
    lift = jump * TORSO * np.exp(-((t - tc - 0.12) / 0.08) ** 2)  # saut : les 2 pieds montent
    ankle_f = np.c_[np.full(T, x0 + split / 2), GROUND - lift]
    ankle_b = np.c_[np.full(T, x0 - split / 2), GROUND - lift]
    hip = np.c_[np.full(T, x0) - 30, GROUND - lift - leg]
    def knee_pos(ankle):
        m, v = (hip + ankle) / 2, hip - ankle
        d = np.linalg.norm(v, axis=1, keepdims=True)
        perp = np.c_[-v[:, 1], v[:, 0]] / d
        perp *= np.sign(perp[:, :1] + 1e-9)  # genou vers l'avant (+x)
        return m + perp * np.sqrt(np.clip(THIGH ** 2 - (d / 2) ** 2, 0, None))

    knee_f, knee_b = knee_pos(ankle_f), knee_pos(ankle_b)
    lean = np.full(T, 35.0) - (_ramp(t, tc - 0.05, tc + 0.2, 0, 25) if rise else 0)
    sh = hip + TORSO * np.c_[np.sin(np.radians(lean)), -np.cos(np.radians(lean))]

    # Bras : formation du plan (-> 70°), puis montée autour du contact (swing)
    tp = tc - 0.45 - arms_early
    phi = _ramp(t, tp - 0.3, tp, 10, 70)
    phi = phi + _ramp(t, tc - 0.2 - arms_early, tc + 0.1 - arms_early, 0, swing)
    down = (hip - sh) / np.linalg.norm(hip - sh, axis=1, keepdims=True)
    arm_dir = _rot(down, -phi)  # rotation vers l'avant (+x) en coordonnées image
    if arm_dir[T // 2, 0] < 0:
        arm_dir = _rot(down, phi)
    elbow = sh + arm_dir * ARM * 0.5
    if bent_elbows:
        elbow = elbow + _rot(arm_dir, np.full(T, 90.0)) * bent_elbows
    wrist = sh + arm_dir * ARM
    nose = sh + np.c_[np.full(T, 60.0), np.full(T, -110.0)]

    xy = np.zeros((T, 17, 2))
    for s, off in (("left", 0.0), ("right", 6.0)):
        xy[:, KP[f"{s}_shoulder"]] = sh + off
        xy[:, KP[f"{s}_elbow"]] = elbow + off
        xy[:, KP[f"{s}_wrist"]] = wrist + off
        xy[:, KP[f"{s}_hip"]] = hip + off
    xy[:, KP["left_knee"]], xy[:, KP["left_ankle"]] = knee_f, ankle_f
    xy[:, KP["right_knee"]], xy[:, KP["right_ankle"]] = knee_b, ankle_b
    for n in ("nose", "left_eye", "right_eye", "left_ear", "right_ear"):
        xy[:, KP[n]] = nose
    xy += rng.normal(0, noise, xy.shape)

    conf = np.full((T, 17, 1), 0.9)
    conf[:, [KP[f"right_{n}"] for n in ("shoulder", "elbow", "wrist", "hip")]] = 0.6  # côté caché

    # Ballon : arrive de l'avant, touche l'avant-bras à forearm_t, repart vers le haut
    ic = int(round(tc * fps))
    p_contact = wrist[ic] + (elbow[ic] - wrist[ic]) * forearm_t + np.array([0, -25.0])
    ball = np.full((T, 3), np.nan)
    if with_ball:
        dt = t - tc
        pre, post = dt < 0, dt >= 0
        ball[pre, 0] = p_contact[0] - dt[pre] * 900
        ball[pre, 1] = p_contact[1] + dt[pre] * 400 + 900 * dt[pre] ** 2 * 0
        ball[post, 0] = p_contact[0] + dt[post] * 250
        ball[post, 1] = p_contact[1] - dt[post] * 1500 + 0.5 * 2500 * dt[post] ** 2
        ball[:, 2] = 0.8
        out = (ball[:, 0] < 0) | (ball[:, 0] > W) | (ball[:, 1] < 0) | (ball[:, 1] > H)
        ball[out] = [np.nan, np.nan, 0]

    kps = np.concatenate([xy, conf], axis=2)
    if facing < 0:
        kps[..., 0] = W - kps[..., 0]
        ball[:, 0] = W - ball[:, 0]
    return PoseResult(kps, ball, np.arange(T), fps, W, H, "synthetic"), ic


FAULTS = {
    "swing": lambda r: dict(swing=r.uniform(40, 80)),
    "saut": lambda r: dict(jump=r.uniform(0.3, 0.7)),
    "redressement": lambda r: dict(rise=True),
    "contact_coude": lambda r: dict(forearm_t=r.uniform(0.8, 1.05)),
    "pieds_alignes": lambda r: dict(split=r.uniform(0, 30)),
    "coudes_plies": lambda r: dict(bent_elbows=r.uniform(60, 120)),
    "bras_en_avance": lambda r: dict(arms_early=r.uniform(0.15, 0.3), swing=r.uniform(20, 40)),
}


def make_dataset(n: int = 80, p_good: float = 0.6, label_noise: float = 0.08, seed: int = 0):
    """Table de features synthétique (une ligne = une manchette), au même format
    que `volleyai extract`. Sert à tester l'entraînement sans vidéos."""
    import pandas as pd

    from volleyai.features.manchette import analyze_pose

    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        good = rng.random() < p_good
        kw = dict(swing=rng.uniform(0, 15), forearm_t=rng.uniform(0.1, 0.5), split=rng.uniform(80, 180),
                  arms_early=rng.uniform(-0.03, 0.05), facing=int(rng.choice([-1, 1])),
                  with_ball=bool(rng.random() < 0.8), noise=rng.uniform(1, 4),
                  t_contact=rng.uniform(1.2, 1.6), seed=i)
        faults = []
        if not good:
            faults = list(rng.choice(list(FAULTS), size=int(rng.integers(1, 3)), replace=False))
            for f in faults:
                kw.update(FAULTS[f](rng))
        pose, _ = make_manchette(**kw)
        _, reps = analyze_pose(pose, {})
        label = int(good) if rng.random() >= label_noise else int(not good)
        rows.append({**reps[0], "label": label, "name": f"SYN_{i // 2:03d}.mov", "file_id": f"syn{i:03d}",
                     "folder_path": "reussie" if label else "ratee", "faults": "|".join(faults),
                     "source_kind": "synthetic"})
    return pd.DataFrame(rows)
