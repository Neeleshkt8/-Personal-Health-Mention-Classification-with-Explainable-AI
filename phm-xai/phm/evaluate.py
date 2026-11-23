"""Metrics, significance tests, error analysis and plots."""
from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,  # noqa: E402
                             roc_auc_score)

from .data import LABELS  # noqa: E402


def metrics(y, proba, y_aux=None, aux_proba=None) -> dict:
    pred = proba.argmax(1)
    rep = classification_report(y, pred, labels=[0, 1, 2], target_names=LABELS, output_dict=True, zero_division=0)
    out = {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "per_class_f1": {l: rep[l]["f1-score"] for l in LABELS},
        "macro_auc_ovr": roc_auc_score(y, proba, multi_class="ovr"),
        "confusion": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
    }
    if aux_proba is not None:
        out["aux_figurative_f1"] = f1_score(y_aux, (aux_proba > 0.5).astype(int))
        out["aux_figurative_auc"] = roc_auc_score(y_aux, aux_proba)
    return out


def mcnemar(y, pred_a, pred_b) -> dict:
    """Exact McNemar test on paired correctness."""
    a, b = pred_a == y, pred_b == y
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"a_right_b_wrong": n01, "a_wrong_b_right": n10, "p_value": p}


def bootstrap_f1_diff(y, pred_a, pred_b, n=1000, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    diffs = []
    for _ in range(n):
        s = rng.choice(idx, len(idx))
        diffs.append(f1_score(y[s], pred_a[s], average="macro") - f1_score(y[s], pred_b[s], average="macro"))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean_diff": float(np.mean(diffs)), "ci95": [float(lo), float(hi)]}


# --------------------------------------------------------------------------- error analysis
_TAGS = {
    "simile_or_metaphor": re.compile(r"\b(like a|like an|feels like|it's like|as if)\b", re.I),
    "near_miss_phrase": re.compile(r"\b(nearly|almost)\b.*\b(heart attack|stroke)\b", re.I),
    "third_person": re.compile(r"\bmy (mom|dad|brother|sister|husband|wife|son|roommate)\b", re.I),
    "question": re.compile(r"\?"),
    "slang_or_emoji": re.compile(r"(lol|lmao|smh|fml|😂|😩|🙃)", re.I),
    "mention_or_url": re.compile(r"(@\w+|http)", re.I),
}


def tag(text: str) -> list[str]:
    t = [k for k, r in _TAGS.items() if r.search(text)]
    n = len(text.split())
    t.append("short(<=6w)" if n <= 6 else "long(>12w)" if n > 12 else "medium")
    return t


def likely_label_noise(y, probas: dict[str, np.ndarray], thresh: float = 0.85) -> np.ndarray:
    """Flag posts where EVERY model confidently agrees on a label different from gold
    (confident-learning style). These are candidates for re-annotation, not model errors."""
    preds = np.stack([p.argmax(1) for p in probas.values()])
    conf = np.stack([p.max(1) for p in probas.values()])
    agree = (preds == preds[0]).all(0)
    return agree & (preds[0] != y) & (conf.min(0) >= thresh)


def error_analysis(test: pd.DataFrame, preds: dict[str, np.ndarray], probas: dict | None = None) -> dict:
    df = test.copy()
    if probas:
        df["likely_mislabeled"] = likely_label_noise(df["label"].to_numpy(), probas)
    df["tags"] = df["text"].map(tag)
    for m, p in preds.items():
        df[f"pred_{m}"] = p
        df[f"ok_{m}"] = p == df["label"].to_numpy()
    rows = []
    for tg in sorted({t for ts in df["tags"] for t in ts}):
        sub = df[df["tags"].map(lambda ts: tg in ts)]
        rows.append({"slice": tg, "n": len(sub), **{f"acc_{m}": round(sub[f"ok_{m}"].mean(), 3) for m in preds}})
    slices = pd.DataFrame(rows).sort_values("n", ascending=False)
    confusions = {}
    for m in preds:
        err = df[~df[f"ok_{m}"]]
        pairs = (err["label"].map(LABELS.__getitem__) + " -> " + err[f"pred_{m}"].map(LABELS.__getitem__))
        confusions[m] = pairs.value_counts().head(5).to_dict()
    names = list(preds)
    disagree = df[df[f"pred_{names[0]}"] != df[f"pred_{names[-1]}"]] if len(names) > 1 else df.iloc[:0]
    noise = {}
    if "likely_mislabeled" in df:
        flagged = df["likely_mislabeled"]
        errs_any = ~df[[f"ok_{m}" for m in preds]].all(1)
        noise = {"flagged": int(flagged.sum()), "share_of_errors": float(flagged.sum() / max(1, errs_any.sum()))}
        if "noisy" in df:  # synthetic data knows which labels were flipped
            tp = int((flagged & (df["noisy"] == 1)).sum())
            noise.update(precision=tp / max(1, flagged.sum()), recall=tp / max(1, (df["noisy"] == 1).sum()))
    return {"frame": df, "slices": slices, "top_confusions": confusions, "disagreements": disagree,
            "label_noise": noise}


# --------------------------------------------------------------------------- plots
def plot_confusions(results: dict, path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8))
    axes = np.atleast_1d(axes)
    short = ["PHM", "OTHER", "FIG"]
    for ax, (m, r) in zip(axes, results.items()):
        cm = np.array(r["confusion"])
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.0%}", ha="center", va="center", fontsize=9,
                        color="white" if cmn[i, j] > 0.6 else "black")
        ax.set_xticks(range(3), short)
        ax.set_yticks(range(3), short)
        ax.set_xlabel("predicted")
        ax.set_ylabel("gold")
        ax.set_title(f"{m}\nmacro-F1 {r['macro_f1']:.3f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_tradeoff(rows: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(rows["infer_ms_per_1k"], rows["macro_f1"], s=40 + 6 * np.sqrt(rows["size_mb"].clip(lower=0.01)) * 10,
               color="#1f77b4", alpha=0.8)
    for _, r in rows.iterrows():
        ax.annotate(r["model"], (r["infer_ms_per_1k"], r["macro_f1"]), textcoords="offset points", xytext=(6, 4),
                    fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("inference time, ms per 1k posts (log)")
    ax.set_ylabel("test macro-F1")
    ax.set_title("Accuracy vs compute (marker size ~ model size)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_top_features(lr_model, path, k=12):
    names, coef = lr_model.feature_names(), lr_model.clf.coef_
    word_mask = np.array([not n.startswith("c:") for n in names])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for c, ax in enumerate(axes):
        w = np.where(word_mask, coef[c], -np.inf)
        top = np.argsort(-w)[:k][::-1]
        ax.barh([names[i] for i in top], coef[c][top], color="#d62728")
        ax.set_title(f"top word features → {LABELS[c]}", fontsize=10)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
