import numpy as np
import pandas as pd
import pytest

from volleyai.synthetic import make_manchette
from volleyai.analysis.reference import build_reference, compare_to_reference
from volleyai.analysis.rules import evaluate, rule_score
from volleyai.config import load_config
from volleyai.features.manchette import FEATURE_COLUMNS, analyze_pose

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


def test_training_registry_end_to_end(tmp_path):
    from volleyai.model.train import list_versions, load_model, predict_proba, promote, train_candidates
    from volleyai.synthetic import make_dataset

    df = make_dataset(n=60, seed=1)
    csv = tmp_path / "features.csv"
    df.to_csv(csv, index=False)
    cfg = {**CFG, "paths": {**CFG["paths"], "models_dir": tmp_path / "models", "reports_dir": tmp_path / "rep"},
           "training": {**CFG["training"], "cv_repeats": 2},
           "mlflow": {**CFG["mlflow"], "tracking_uri": f"sqlite:///{tmp_path}/mlflow.db",
                      "experiment": "test"}}
    res = train_candidates(csv, cfg)
    board = res["leaderboard"]
    assert set(board["candidate"]) == set(cfg["training"]["candidates"])
    assert board["balanced_accuracy_mean"].max() >= 0.75, board
    assert res["registered"]["promoted"]  # premier modèle -> champion

    res2 = train_candidates(csv, cfg, candidates=["logreg"])
    assert not res2["registered"]["promoted"]  # le champion n'est pas écrasé
    versions = list_versions(cfg)
    assert [v["version"] for v in versions] == [1, 2]
    assert "champion" in versions[0]["aliases"] and "challenger" in versions[1]["aliases"]

    weak = {**cfg, "training": {**cfg["training"], "min_score_to_serve": 0.99}}
    from volleyai.model.train import demote
    demote(cfg)
    assert load_model(cfg) is None  # sans champion : règles du coach
    assert not train_candidates(csv, weak, candidates=["logreg"])["registered"]["promoted"]
    promote(cfg, 2)
    bundle = load_model(cfg)
    assert bundle["meta"]["version"] == 2
    good = df[df["label"] == 1].iloc[0].to_dict()
    assert 0.0 <= predict_proba(bundle, good) <= 1.0

    ref = build_reference(df)
    bad, _ = feats(swing=75, seed=99)
    assert any(d["feature"].startswith("arm") for d in compare_to_reference(bad, ref, top=8))


def test_multiple_reps_segmented():
    from volleyai.pose import PoseResult
    a, _ = make_manchette(seed=1)
    b, _ = make_manchette(seed=2, swing=70)
    pose = PoseResult(np.concatenate([a.keypoints, b.keypoints]), np.concatenate([a.ball, b.ball]),
                      np.arange(len(a.keypoints) * 2), a.fps, a.width, a.height, "x")
    _, reps = analyze_pose(pose, CFG)
    assert len(reps) == 2
