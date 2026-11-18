import numpy as np
import pandas as pd
import pytest

from synthetic import make_manchette
from volleyai.analysis.reference import build_reference, compare_to_reference
from volleyai.analysis.rules import evaluate, rule_score
from volleyai.config import load_config
from volleyai.features.manchette import FEATURE_COLUMNS, analyze_pose
from volleyai.model.train import cross_validate, make_pipeline

CFG = load_config()


def feats(**kw):
    pose, ic = make_manchette(**kw)
    _, reps = analyze_pose(pose, CFG)
    assert len(reps) == 1, reps
    return reps[0], ic


def status(f):
    return {r.id: r.status for r in evaluate(f, CFG["criteria"])}


def test_contact_detected_with_ball():
    f, ic = feats()
    assert f["contact_method"] == "ball"
    assert abs(f["contact_index"] - ic) <= 3


def test_contact_detected_without_ball():
    f, ic = feats(with_ball=False)
    assert f["contact_method"] == "pose"
    assert abs(f["contact_index"] - ic) <= int(0.3 * 60)


def test_good_manchette_passes_rules():
    f, _ = feats()
    st = status(f)
    assert all(v == "ok" for v in st.values()), (st, {k: f[k] for k in FEATURE_COLUMNS})
    assert rule_score(evaluate(f, CFG["criteria"])) == 1.0


@pytest.mark.parametrize("kw, failing", [
    (dict(swing=70), "pas_de_swing"),
    (dict(jump=0.6), "pas_de_saut"),
    (dict(rise=True), "pas_de_remontee"),
    (dict(forearm_t=0.95), "contact_avant_bras"),
    (dict(split=10), "placement_jambes"),
    (dict(bent_elbows=90), "bras_tendus"),
])
def test_faults_are_detected(kw, failing):
    f, _ = feats(**kw)
    assert status(f)[failing] == "ko", {k: f[k] for k in FEATURE_COLUMNS}


def test_mirror_invariance():
    a, _ = feats(facing=1)
    b, _ = feats(facing=-1)
    for k in ("elbow_angle_contact_deg", "arm_swing_amplitude_deg", "rise_contact", "stance_split"):
        assert a[k] == pytest.approx(b[k], abs=1e-6), k


def test_no_ball_criterion_not_evaluable():
    f, _ = feats(with_ball=False)
    assert status(f)["contact_avant_bras"] == "na"


def _dataset(n=24, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        good = i % 3 != 0
        kw = dict(swing=rng.uniform(0, 15), noise=2, seed=i) if good else \
            dict(swing=rng.uniform(50, 80), jump=rng.choice([0, 0.5]), noise=2, seed=i)
        f, _ = feats(**kw)
        rows.append({**f, "label": int(good), "name": f"clip{i}.mov"})
    return pd.DataFrame(rows)


def test_training_and_reference_end_to_end():
    df = _dataset()
    cv = cross_validate(df)
    assert cv["balanced_accuracy"] >= 0.8, cv
    model = make_pipeline().fit(df[FEATURE_COLUMNS].to_numpy(float), df["label"])
    assert model.predict_proba(df[FEATURE_COLUMNS].to_numpy(float)).shape == (len(df), 2)
    ref = build_reference(df)
    assert next(iter(ref["features"])) in ("arm_swing_amplitude_deg", "arm_peak_velocity_dps",
                                           "arm_contribution", "legs_arms_lag_s", "legs_arms_lag_abs_s",
                                           "arm_trunk_contact_deg")
    bad, _ = feats(swing=75, seed=99)
    devs = compare_to_reference(bad, ref)
    assert any(d["feature"].startswith("arm") for d in devs)


def test_multiple_reps_segmented():
    from volleyai.pose import PoseResult
    a, _ = make_manchette(seed=1)
    b, _ = make_manchette(seed=2, swing=70)
    pose = PoseResult(np.concatenate([a.keypoints, b.keypoints]), np.concatenate([a.ball, b.ball]),
                      np.arange(len(a.keypoints) * 2), a.fps, a.width, a.height, "x")
    _, reps = analyze_pose(pose, CFG)
    assert len(reps) == 2
