"""Explainability.

* linear_contributions  - exact attributions for TF-IDF+LogReg (feature value x class coefficient),
                          aggregated from word/char n-grams back to words.
* occlusion             - model-agnostic: drop each word, measure change in p(target).
* lime_lite             - model-agnostic LIME-style local surrogate: random word masks,
                          cosine-kernel-weighted ridge regression (no `lime` dependency).
* faithfulness          - comprehensiveness test: removing the top-k attributed words should
                          hurt p(target) more than removing k random words.
* DistilBERT also has Integrated Gradients (distilbert_mtl.py).
"""
from __future__ import annotations

import html

import numpy as np
from sklearn.linear_model import Ridge

from .data import LABELS
from .text import clean


def _words(text: str) -> list[str]:
    return clean(text).split()


def linear_contributions(model, text: str, target: int) -> list[tuple[str, float]]:
    words = _words(text)
    x = model.features([text]).tocoo()
    names = model.feature_names()
    coef = model.clf.coef_[target]
    scores = np.zeros(len(words))
    for j, val in zip(x.col, x.data):
        contrib = val * coef[j]
        f = names[j]
        if f.startswith("c:"):  # char n-gram: credit every word whose padded form contains it
            g = f[2:]
            hits = [i for i, w in enumerate(words) if g in f" {w} "]
        else:
            toks = f.split()
            hits = [i for i in range(len(words) - len(toks) + 1) if words[i : i + len(toks)] == toks]
            hits = [i + k for i in hits for k in range(len(toks))]
        for i in hits:
            scores[i] += contrib / len(hits)
    return list(zip(words, scores.tolist()))


def occlusion(predict_proba, text: str, target: int) -> list[tuple[str, float]]:
    words = _words(text)
    variants = [" ".join(words)] + [" ".join(words[:i] + words[i + 1 :]) for i in range(len(words))]
    p = predict_proba(variants)[:, target]
    return list(zip(words, (p[0] - p[1:]).tolist()))


def lime_lite(predict_proba, text: str, target: int, n_samples: int = 400, seed: int = 0, kernel_width=0.25):
    words = _words(text)
    d = len(words)
    rng = np.random.default_rng(seed)
    Z = (rng.random((n_samples, d)) > rng.uniform(0.1, 0.6, (n_samples, 1))).astype(float)
    Z[0] = 1
    texts = [" ".join(w for w, keep in zip(words, z) if keep) for z in Z]
    y = predict_proba(texts)[:, target]
    cos = Z.sum(1) / (np.sqrt(Z.sum(1)) * np.sqrt(d) + 1e-12)
    w = np.exp(-((1 - cos) ** 2) / kernel_width**2)
    reg = Ridge(alpha=1.0).fit(Z, y, sample_weight=w)
    return list(zip(words, reg.coef_.tolist()))


def faithfulness(predict_proba, texts, explainer, k: int = 2, seed: int = 0) -> dict:
    """Mean drop in p(predicted class) when deleting the top-k attributed words vs k random words."""
    rng = np.random.default_rng(seed)
    top_drops, rand_drops = [], []
    for t in texts:
        words = _words(t)
        if len(words) <= k:
            continue
        p0 = predict_proba([" ".join(words)])[0]
        c = int(p0.argmax())
        attr = np.array([s for _, s in explainer(t, c)])
        top = set(np.argsort(-attr)[:k])
        rnd = set(rng.choice(len(words), k, replace=False))
        without = lambda S: " ".join(w for i, w in enumerate(words) if i not in S)  # noqa: E731
        p = predict_proba([without(top), without(rnd)])[:, c]
        top_drops.append(p0[c] - p[0])
        rand_drops.append(p0[c] - p[1])
    return {"k": k, "n": len(top_drops), "top_k_drop": float(np.mean(top_drops)),
            "random_k_drop": float(np.mean(rand_drops))}


# --------------------------------------------------------------------------- rendering
def _span(word, score, scale):
    a = min(1.0, abs(score) / scale) if scale else 0
    color = f"rgba(214,39,40,{a:.2f})" if score > 0 else f"rgba(31,119,180,{a:.2f})"
    return f'<span title="{score:+.3f}" style="background:{color};padding:1px 3px;border-radius:3px">{html.escape(word)}</span>'


def render_row(attrs: list[tuple[str, float]]) -> str:
    scale = max((abs(s) for _, s in attrs), default=0)
    return " ".join(_span(w, s, scale) for w, s in attrs)


def render_report(cases: list[dict], path) -> None:
    """cases: [{text, gold, preds: {model: (label, prob)}, explanations: {method: attrs}}]"""
    parts = ["<html><head><meta charset='utf-8'><title>PHM explanations</title><style>"
             "body{font-family:system-ui,sans-serif;max-width:1000px;margin:2em auto;color:#222}"
             ".case{border:1px solid #ddd;border-radius:8px;padding:12px 16px;margin:14px 0}"
             "td{padding:4px 8px;vertical-align:top}.m{color:#666;font-size:13px;white-space:nowrap}"
             ".bad{color:#c0392b;font-weight:600}.ok{color:#27ae60;font-weight:600}</style></head><body>",
             "<h1>Health-mention classifier explanations</h1>",
             "<p><span style='background:rgba(214,39,40,.6);padding:1px 4px'>red</span> pushes towards the "
             "explained class, <span style='background:rgba(31,119,180,.6);padding:1px 4px'>blue</span> pushes "
             "away. Hover a word for its score.</p>"]
    for c in cases:
        preds = " · ".join(
            f"{m}: <span class='{'ok' if lab == c['gold'] else 'bad'}'>{lab}</span> ({p:.2f})"
            for m, (lab, p) in c["preds"].items())
        rows = "".join(f"<tr><td class='m'>{html.escape(k)}</td><td>{render_row(v)}</td></tr>"
                       for k, v in c["explanations"].items())
        parts.append(f"<div class='case'><div><b>Gold:</b> {c['gold']} &nbsp; {preds}</div>"
                     f"<div class='m'>{html.escape(c.get('note', ''))}</div><table>{rows}</table></div>")
    parts.append("</body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


__all__ = ["linear_contributions", "occlusion", "lime_lite", "faithfulness", "render_report", "LABELS"]
