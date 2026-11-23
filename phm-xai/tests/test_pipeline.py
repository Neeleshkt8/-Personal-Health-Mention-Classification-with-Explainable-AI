import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from phm import evaluate as ev
from phm import explain as xp
from phm.data import load_csv, make_synthetic, split
from phm.models import MultiTaskMLP, TfidfLogReg

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def data():
    return split(make_synthetic(1200, seed=1))


@pytest.fixture(scope="module")
def lr(data):
    return TfidfLogReg().fit(data[0], data[1])


@pytest.fixture(scope="module")
def mtl(data):
    return MultiTaskMLP(lambda_aux=0.5, epochs=10).fit(data[0], data[1])


# ---------------- data
def test_synthetic_is_deterministic_and_valid():
    a, b = make_synthetic(500, seed=3), make_synthetic(500, seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["label"]) == {0, 1, 2} and set(a["figurative"]) <= {0, 1}
    assert (a.loc[a["label"] == 2, "figurative"] == 1).all()          # figurative class => aux=1
    assert a.loc[a["label"] == 0, "figurative"].mean() > 0            # aux is not just a copy of the label


def test_split_is_stratified_and_disjoint(data):
    tr, va, te = data
    assert not set(tr["text"]) & set(te["text"])
    assert abs(tr["label"].mean() - te["label"].mean()) < 0.1


def test_load_csv_maps_label_names(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"text": ["i have the flu", "flu season stats", "this exam is giving me a migraine"],
                  "label": ["personal", "non-personal", "figurative"]}).to_csv(p, index=False)
    df = load_csv(p)
    assert df["label"].tolist() == [0, 1, 2] and df["figurative"].tolist() == [0, 0, 1]
    pd.DataFrame({"text": ["x"], "label": ["???"]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_csv(p)


# ---------------- models
def test_models_learn(data, lr, mtl):
    te = data[2]
    for m in (lr, mtl):
        p = m.predict_proba(te["text"])
        assert p.shape == (len(te), 3) and np.allclose(p.sum(1), 1, atol=1e-5)
        assert (p.argmax(1) == te["label"]).mean() > 0.85
    aux = mtl.predict_aux(te["text"])
    assert ev.metrics(te["label"].to_numpy(), mtl.predict_proba(te["text"]), te["figurative"].to_numpy(), aux)[
        "aux_figurative_f1"] > 0.8
    assert lr.predict_aux(te["text"]) is None


def test_mtl_gradients_match_numerical():
    rng = np.random.default_rng(0)
    m = MultiTaskMLP(hidden=5, lambda_aux=0.7, l2=1e-3)
    d, n = 7, 6
    m.params = [rng.normal(0, .5, (d, 5)), rng.normal(0, .1, 5), rng.normal(0, .5, (5, 3)), rng.normal(0, .1, 3),
                rng.normal(0, .5, (5, 1)), rng.normal(0, .1, 1)]
    X = sparse.csr_matrix(rng.random((n, d)))
    Y = np.eye(3)[rng.integers(0, 3, n)]
    ya = rng.integers(0, 2, n).astype(float)
    _, grads = m._loss_and_grads(X, Y, ya)
    for k, P in enumerate(m.params):
        num = np.zeros_like(P)
        for idx in np.ndindex(P.shape):
            old = P[idx]
            P[idx] = old + 1e-5
            lp, _ = m._loss_and_grads(X, Y, ya)
            P[idx] = old - 1e-5
            lm, _ = m._loss_and_grads(X, Y, ya)
            P[idx] = old
            num[idx] = (lp - lm) / 2e-5
        assert np.allclose(np.asarray(grads[k]).reshape(P.shape), num, atol=1e-5), f"param {k}"


# ---------------- evaluation
def test_mcnemar_and_bootstrap():
    y = np.array([0, 1, 2] * 20)
    a, b = y.copy(), y.copy()
    b[:10] = (b[:10] + 1) % 3
    r = ev.mcnemar(y, a, b)
    assert r["a_right_b_wrong"] == 10 and r["p_value"] < 0.01
    assert ev.bootstrap_f1_diff(y, a, b, n=200)["ci95"][0] > 0


def test_label_noise_detector():
    y = np.array([0, 1, 2, 0])
    conf_wrong = np.array([[.95, .03, .02], [0, 1, 0], [0, 0, 1], [.05, .05, .9]])
    assert ev.likely_label_noise(y, {"a": conf_wrong, "b": conf_wrong}).tolist() == [False, False, False, True]


# ---------------- explainability
def test_linear_contributions_point_the_right_way(lr):
    t = "this group project is giving me a migraine lol"
    attrs = dict(xp.linear_contributions(lr, t, 2))
    assert attrs["project"] > 0                     # 'project' evidence for FIGURATIVE


def test_explanations_are_faithful(lr, mtl, data):
    texts = data[2]["text"].head(60).tolist()
    for m in (lr, mtl):
        f = xp.faithfulness(m.predict_proba, texts, lambda t, c, m=m: xp.occlusion(m.predict_proba, t, c))
        assert f["top_k_drop"] > f["random_k_drop"]
    a = xp.lime_lite(mtl.predict_proba, texts[0], 0, n_samples=100)
    assert len(a) == len(texts[0].lower().split()) or len(a) > 0


def test_html_render(tmp_path):
    xp.render_report([{"text": "x", "gold": "FIGURATIVE", "preds": {"m": ("FIGURATIVE", .9)},
                       "explanations": {"occ": [("<b>", 0.5), ("y", -0.2)]}}], tmp_path / "e.html")
    assert "&lt;b&gt;" in (tmp_path / "e.html").read_text()   # escaped


# ---------------- end to end
def test_pipeline_end_to_end(tmp_path):
    r = subprocess.run([sys.executable, "run_pipeline.py", "--n-synthetic", "800", "--seeds", "0",
                        "--n-explain", "4", "--out", str(tmp_path)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    for f in ["report.md", "summary.json", "explanations.html", "confusion_matrices.png",
              "accuracy_vs_compute.png", "test_predictions.csv", "error_slices.csv"]:
        assert (tmp_path / f).exists(), f


def test_distilbert_module_importable():
    from phm import distilbert_mtl
    if not distilbert_mtl.TORCH_OK:
        with pytest.raises(ImportError):
            distilbert_mtl.DistilBertMTL()
