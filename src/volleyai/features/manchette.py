"""Features biomécaniques de la manchette (couche Gold).

Hypothèses : vue de profil, un joueur principal. Après orientation, le joueur
regarde toujours vers +x (le ballon arrive de +x). Les hauteurs sont exprimées
vers le haut et normalisées par la longueur du tronc (épaules-hanches), ce qui
rend les features indépendantes du zoom et de la distance caméra.

Chaque vidéo peut contenir plusieurs manchettes : on détecte les contacts,
puis on calcule un vecteur de features par répétition.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from volleyai.keypoints import angle, clean_keypoints, facing_sign, interpolate, mirror_x, near_side
from volleyai.pose import KP, PoseResult

# Colonnes utilisées par le modèle (l'ordre compte pour le modèle sauvegardé)
FEATURE_COLUMNS = [
    "elbow_angle_contact_deg",
    "knee_angle_contact_deg",
    "knee_angle_min_prep_deg",
    "trunk_lean_contact_deg",
    "arm_trunk_contact_deg",
    "arm_swing_amplitude_deg",
    "arm_peak_velocity_dps",
    "arm_contribution",
    "rise_contact",
    "leg_drive_pre",
    "ankle_lift_contact",
    "legs_arms_lag_s",
    "legs_arms_lag_abs_s",
    "stance_split",
    "wrist_forward_contact",
    "contact_forearm_t",
    "contact_forearm_dist",
]

FEATURE_DESCRIPTIONS = {
    "elbow_angle_contact_deg": "Angle du coude au contact (180° = bras tendu)",
    "knee_angle_contact_deg": "Angle du genou au contact (180° = jambe tendue)",
    "knee_angle_min_prep_deg": "Flexion maximale du genou en préparation",
    "trunk_lean_contact_deg": "Inclinaison du buste vers l'avant au contact",
    "arm_trunk_contact_deg": "Angle bras/tronc au contact",
    "arm_swing_amplitude_deg": "Montée des bras autour du contact (swing)",
    "arm_peak_velocity_dps": "Vitesse max de montée des bras (°/s)",
    "arm_contribution": "Montée des poignets non expliquée par les jambes (tronc)",
    "rise_contact": "Redressement du haut du corps (épaules) autour du contact (tronc)",
    "leg_drive_pre": "Poussée des jambes avant le contact (tronc)",
    "ankle_lift_contact": "Décollement des deux pieds au contact (tronc)",
    "legs_arms_lag_s": "Décalage poussée jambes -> montée bras (s, <0 = jambes d'abord)",
    "legs_arms_lag_abs_s": "Décalage absolu jambes/bras (s)",
    "stance_split": "Écart avant/arrière des pieds (tronc)",
    "wrist_forward_contact": "Avancée des poignets devant les épaules (tronc)",
    "contact_forearm_t": "Position du ballon sur l'avant-bras (0 = poignet, 1 = coude)",
    "contact_forearm_dist": "Distance ballon / avant-bras au contact (tronc)",
}


@dataclass
class Series:
    fps: float
    side: str
    facing: int
    torso: float
    xy: np.ndarray           # (T, 17, 2) orienté, pixels
    ball: np.ndarray         # (T, 2) orienté, pixels (NaN si absent)
    ball_conf: np.ndarray    # (T,)
    quality: np.ndarray      # (T,) confiance moyenne des points utiles
    frame_indices: np.ndarray
    # séries dérivées
    up: dict                 # hauteurs normalisées (vers le haut)
    knee: np.ndarray
    elbow: np.ndarray
    arm_trunk: np.ndarray
    trunk_lean: np.ndarray
    wrist_fwd: np.ndarray
    stance: np.ndarray


def _j(side: str, name: str) -> int:
    return KP[f"{side}_{name}"]


def build_series(pose: PoseResult, conf_min: float = 0.3, smoothing_window_s: float = 0.12) -> Series:
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*(All-NaN|Mean of empty).*")
    kps = pose.keypoints
    fps = pose.fps
    side = near_side(kps)
    xy = clean_keypoints(kps, fps, conf_min, smoothing_window_s)
    facing = facing_sign(xy)
    ball = pose.ball[:, :2].copy()
    if facing < 0:
        xy = mirror_x(xy, pose.width)
        ball = mirror_x(ball, pose.width)

    sh = (xy[:, KP["left_shoulder"]] + xy[:, KP["right_shoulder"]]) / 2
    hip = (xy[:, KP["left_hip"]] + xy[:, KP["right_hip"]]) / 2
    torso = float(np.nanmedian(np.linalg.norm(sh - hip, axis=1)))
    if not np.isfinite(torso) or torso <= 1:
        torso = float(pose.height) / 4  # repli grossier

    y_ref = np.nanmedian(xy[:, [KP["left_ankle"], KP["right_ankle"]], 1])

    def up(p: np.ndarray) -> np.ndarray:
        return -(p[:, 1] - y_ref) / torso

    S, E, W = xy[:, _j(side, "shoulder")], xy[:, _j(side, "elbow")], xy[:, _j(side, "wrist")]
    H, K, A = xy[:, _j(side, "hip")], xy[:, _j(side, "knee")], xy[:, _j(side, "ankle")]

    trunk_vec = sh - hip
    trunk_lean = np.degrees(np.arctan2(trunk_vec[:, 0], -trunk_vec[:, 1]))  # >0 = penché vers l'avant

    useful = [_j(side, n) for n in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")]
    quality = np.nan_to_num(kps[:, useful, 2]).mean(axis=1)

    return Series(
        fps=fps, side=side, facing=facing, torso=torso, xy=xy, ball=ball,
        ball_conf=np.nan_to_num(pose.ball[:, 2]), quality=quality,
        frame_indices=pose.frame_indices,
        up={
            "hip": up(hip), "shoulder": up(sh), "wrist": up(W),
            "ankle_l": up(xy[:, KP["left_ankle"]]), "ankle_r": up(xy[:, KP["right_ankle"]]),
        },
        knee=angle(H, K, A),
        elbow=angle(S, E, W),
        arm_trunk=angle(H, S, W),
        trunk_lean=trunk_lean,
        wrist_fwd=(W[:, 0] - S[:, 0]) / torso,
        stance=np.abs(xy[:, KP["left_ankle"], 0] - xy[:, KP["right_ankle"], 0]) / torso,
    )


# --------------------------------------------------------------------------
# Détection des contacts
# --------------------------------------------------------------------------
def _segment_point_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance de p au segment [a, b] et position t (0 = a, 1 = b), vectorisé."""
    ab = b - a
    t = ((p - a) * ab).sum(-1) / ((ab * ab).sum(-1) + 1e-9)
    proj = a + np.clip(t, 0, 1)[:, None] * ab
    return np.linalg.norm(p - proj, axis=-1), t


def forearm_ball(s: Series) -> tuple[np.ndarray, np.ndarray]:
    """Distance ballon/avant-bras (en troncs) et position t (0 poignet -> 1 coude)."""
    W, E = s.xy[:, _j(s.side, "wrist")], s.xy[:, _j(s.side, "elbow")]
    d, t = _segment_point_distance(s.ball, W, E)
    return d / s.torso, t


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i)); start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def fill_short_gaps(arr: np.ndarray, max_gap: int) -> np.ndarray:
    """Interpole les trous (NaN) de longueur <= max_gap, laisse les autres en NaN."""
    missing = np.isnan(arr[:, 0])
    full = interpolate(arr)
    out = arr.copy()
    for a, b in _runs(missing):
        if a > 0 and b < len(arr) and b - a <= max_gap:
            out[a:b] = full[a:b]
    return out


def detect_contacts(s: Series, min_gap_s: float = 1.0) -> list[tuple[int, str]]:
    """Renvoie [(index, méthode)] ; méthode = 'ball' ou 'pose'."""
    T = len(s.knee)
    gap = max(1, int(round(min_gap_s * s.fps)))
    contacts: list[tuple[int, str]] = []

    # 1) Avec le ballon : minimum local de la distance ballon / avant-bras
    has_ball = (~np.isnan(s.ball[:, 0])).sum()
    if has_ball >= max(5, 0.1 * s.fps):
        tmp = Series(**{**s.__dict__, "ball": fill_short_gaps(s.ball, int(0.1 * s.fps))})
        d, _ = forearm_ball(tmp)
        d = np.where(np.isnan(d), np.inf, d)
        for i in np.argsort(d):
            if not np.isfinite(d[i]) or d[i] > 0.6:
                break
            if all(abs(i - c) >= gap for c, _ in contacts):
                contacts.append((int(i), "ball"))
    if contacts:
        return sorted(contacts)

    # 2) Sans ballon : segments "plan de frappe" (bras tendus, levés, devant)
    platform = (s.elbow > 140) & (s.arm_trunk > 35) & (s.arm_trunk < 130) & (s.wrist_fwd > 0.3)
    min_len = max(2, int(0.08 * s.fps))
    for a, b in _runs(platform):
        if b - a < min_len:
            continue
        i = a + int(np.nanargmax(s.wrist_fwd[a:b]))
        if all(abs(i - c) >= gap for c, _ in contacts):
            contacts.append((i, "pose"))
    if contacts:
        return sorted(contacts)

    # 3) Rien de concluant : extension maximale des bras
    return [(int(np.nanargmax(np.nan_to_num(s.wrist_fwd, nan=-9))), "fallback")]


# --------------------------------------------------------------------------
# Features autour d'un contact
# --------------------------------------------------------------------------
def _win(c: int, a_s: float, b_s: float, fps: float, T: int) -> slice:
    return slice(max(0, c + int(round(a_s * fps))), min(T, c + int(round(b_s * fps)) + 1))


def _nan(f, x, default=np.nan):
    x = np.asarray(x, dtype=float)
    return float(f(x)) if np.isfinite(x).any() else default


def extract_features(s: Series, c: int, method: str) -> dict:
    T, fps = len(s.knee), s.fps
    W = lambda a, b: _win(c, a, b, fps, T)  # noqa: E731
    at = lambda arr: arr[c]  # noqa: E731

    f: dict[str, float | str | int] = {}
    f["elbow_angle_contact_deg"] = _nan(np.nanmedian, s.elbow[W(-0.03, 0.03)])
    f["knee_angle_contact_deg"] = float(at(s.knee))
    f["knee_angle_min_prep_deg"] = _nan(np.nanmin, s.knee[W(-0.6, 0.0)])
    f["trunk_lean_contact_deg"] = float(at(s.trunk_lean))
    f["arm_trunk_contact_deg"] = float(at(s.arm_trunk))

    w = W(-0.15, 0.3)
    f["arm_swing_amplitude_deg"] = max(0.0, _nan(np.nanmax, s.arm_trunk[w]) - s.arm_trunk[w.start])
    vel_arm = np.gradient(s.arm_trunk) * fps
    f["arm_peak_velocity_dps"] = _nan(np.nanmax, vel_arm[W(-0.15, 0.2)])

    w = W(-0.15, 0.2)
    wrist_rise = _nan(np.nanmax, s.up["wrist"][w]) - s.up["wrist"][w.start]
    hip_rise = _nan(np.nanmax, s.up["hip"][w]) - s.up["hip"][w.start]
    f["arm_contribution"] = wrist_rise - hip_rise

    w = W(-0.1, 0.3)
    # se redresser = extension des genoux + redressement du buste -> on suit les épaules
    f["rise_contact"] = max(0.0, _nan(np.nanmax, s.up["shoulder"][w]) - s.up["shoulder"][w.start])
    f["leg_drive_pre"] = float(at(s.up["hip"]) - _nan(np.nanmin, s.up["hip"][W(-0.5, 0.0)]))

    lower_foot = np.minimum(s.up["ankle_l"], s.up["ankle_r"])  # >0 seulement si les 2 pieds montent
    base = _nan(np.nanmedian, lower_foot[W(-0.6, -0.2)], default=_nan(np.nanmedian, lower_foot))
    f["ankle_lift_contact"] = max(0.0, _nan(np.nanmax, lower_foot[W(-0.1, 0.3)]) - base)

    # Synchro : instant du pic de poussée des hanches vs pic de montée des bras.
    # Si les bras ne bougent presque pas, il n'y a pas de désynchronisation.
    vel_hip = np.gradient(s.up["hip"]) * fps
    wh, wa = W(-0.4, 0.1), W(-0.25, 0.2)
    if f["arm_swing_amplitude_deg"] < 5:
        lag = 0.0
    elif np.isfinite(vel_hip[wh]).any() and np.isfinite(vel_arm[wa]).any():
        lag = ((wh.start + np.nanargmax(vel_hip[wh])) - (wa.start + np.nanargmax(vel_arm[wa]))) / fps
    else:
        lag = np.nan
    f["legs_arms_lag_s"] = float(lag)
    f["legs_arms_lag_abs_s"] = float(abs(lag))

    f["stance_split"] = _nan(np.nanmedian, s.stance[W(-0.05, 0.05)])
    f["wrist_forward_contact"] = float(at(s.wrist_fwd))

    filled = Series(**{**s.__dict__, "ball": fill_short_gaps(s.ball, int(0.1 * fps))})
    if method == "ball" and not np.isnan(filled.ball[c, 0]):
        d, t = forearm_ball(filled)
        f["contact_forearm_dist"] = float(d[c])
        f["contact_forearm_t"] = float(t[c])
    else:
        f["contact_forearm_dist"] = np.nan
        f["contact_forearm_t"] = np.nan

    # méta-données (pas des features du modèle)
    f["contact_index"] = int(c)
    f["contact_frame"] = int(s.frame_indices[c])
    f["contact_time_s"] = round(float(s.frame_indices[c]) / (fps * max(1, _stride(s))), 3)
    f["contact_method"] = method
    f["camera_side"] = s.side
    f["pose_quality"] = _nan(np.nanmean, s.quality[W(-0.5, 0.3)], default=0.0)
    return f


def _stride(s: Series) -> int:
    if len(s.frame_indices) < 2:
        return 1
    return int(max(1, np.median(np.diff(s.frame_indices))))


def analyze_pose(pose: PoseResult, cfg: dict | None = None) -> tuple[Series, list[dict]]:
    """PoseResult -> (séries, [features par répétition])."""
    cfg = cfg or {}
    pcfg, scfg = cfg.get("pose", {}), cfg.get("segmentation", {})
    s = build_series(pose, pcfg.get("keypoint_conf_min", 0.3), pcfg.get("smoothing_window_s", 0.12))
    reps = []
    for k, (c, method) in enumerate(detect_contacts(s, scfg.get("min_gap_s", 1.0))):
        feats = extract_features(s, c, method)
        feats["rep"] = k
        reps.append(feats)
    return s, reps
