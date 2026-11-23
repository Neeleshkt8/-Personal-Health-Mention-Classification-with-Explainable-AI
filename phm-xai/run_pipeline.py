"""End-to-end pipeline: data -> train/compare models -> error analysis -> explainability -> report.

  python run_pipeline.py                                  # synthetic demo data, offline models
  python run_pipeline.py --data my.csv --distilbert       # real data + DistilBERT multi-task
  python run_pipeline.py --seeds 0 1 2                    # repeat over seeds, report mean ± std
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from phm import evaluate as ev
from phm import explain as xp
from phm.data import LABELS, load_csv, make_synthetic, split
from phm.models import MultiTaskMLP, TfidfLogReg, inference_ms_per_1k, timed_fit

log = logging.getLogger("phm")


def build_models(args, seed):
    ms = [TfidfLogReg(),
          MultiTaskMLP(lambda_aux=0.0, seed=seed),             # single-task ablation
          MultiTaskMLP(lambda_aux=args.lambda_aux, seed=seed)]  # multi-task
    if args.distilbert:
        from phm.distilbert_mtl import TORCH_OK, DistilBertMTL
        if TORCH_OK:
            ms.append(DistilBertMTL(lambda_aux=args.lambda_aux, epochs=args.bert_epochs, seed=seed))
            if args.bert_ablation:
                ms.append(DistilBertMTL(lambda_aux=0.0, epochs=args.bert_epochs, seed=seed))
        else:
            log.warning("--distilbert requested but torch/transformers not installed; skipping")
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="CSV with text,label[,figurative]; omit for synthetic demo data")
    ap.add_argument("--n-synthetic", type=int, default=3000)
    ap.add_argument("--distilbert", action="store_true")
    ap.add_argument("--bert-ablation", action="store_true", help="also train single-task DistilBERT")
    ap.add_argument("--bert-epochs", type=int, default=3)
    ap.add_argument("--lambda-aux", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-explain", type=int, default=12)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- data
    df = load_csv(args.data) if args.data else make_synthetic(args.n_synthetic)
    source = args.data or f"synthetic (n={len(df)})"
    log.info("data: %s | label dist %s | figurative rate %.2f", source,
             df["label"].value_counts().sort_index().tolist(), df["figurative"].mean())

    # ---------------- train / evaluate over seeds
    per_seed, keep = [], {}
    for si, seed in enumerate(args.seeds):
        train, val, test = split(df, seed=seed)
        for m in build_models(args, seed):
            fit_s = timed_fit(m, train, val)
            proba = m.predict_proba(test["text"])
            aux = m.predict_aux(test["text"])
            r = ev.metrics(test["label"].to_numpy(), proba, test["figurative"].to_numpy(), aux)
            r.update(model=m.name, seed=seed, train_s=fit_s, infer_ms_per_1k=inference_ms_per_1k(m, test["text"]),
                     n_params=m.n_params, size_mb=m.size_bytes / 1e6)
            per_seed.append(r)
            log.info("seed %s %-15s macroF1=%.3f acc=%.3f train=%.1fs", seed, m.name, r["macro_f1"],
                     r["accuracy"], fit_s)
            if si == 0:
                keep[m.name] = (m, proba, r)
        if si == 0:
            test0 = test

    res = pd.DataFrame(per_seed)
    agg = res.groupby("model", sort=False).agg(
        macro_f1=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"), accuracy=("accuracy", "mean"),
        auc=("macro_auc_ovr", "mean"), aux_fig_f1=("aux_figurative_f1", "mean"), train_s=("train_s", "mean"),
        infer_ms_per_1k=("infer_ms_per_1k", "mean"), n_params=("n_params", "first"), size_mb=("size_mb", "first"),
    ).reset_index()
    agg["macro_f1_std"] = agg["macro_f1_std"].fillna(0)
    per_class = {m: {l: float(np.mean([d[l] for d in g["per_class_f1"]])) for l in LABELS}
                 for m, g in res.groupby("model", sort=False)}

    # ---------------- significance (seed-0 split)
    y = test0["label"].to_numpy()
    preds = {k: v[1].argmax(1) for k, v in keep.items()}
    names = list(preds)
    sig = {}
    for a, b in [("mtl_mlp", "stl_mlp"), ("mtl_mlp", "tfidf_logreg"), ("distilbert_mtl", "tfidf_logreg"),
                 ("distilbert_mtl", "mtl_mlp"), ("distilbert_mtl", "distilbert_stl")]:
        if a in preds and b in preds:
            sig[f"{a} vs {b}"] = {**ev.mcnemar(y, preds[a], preds[b]),
                                  "f1_diff": ev.bootstrap_f1_diff(y, preds[a], preds[b])}

    # ---------------- error analysis
    ea = ev.error_analysis(test0, preds, {k: v[1] for k, v in keep.items()})
    ea["slices"].to_csv(out / "error_slices.csv", index=False)
    ea["frame"].assign(label=lambda d: d["label"].map(LABELS.__getitem__)).to_csv(out / "test_predictions.csv",
                                                                               index=False)
    ev.plot_confusions({k: v[2] for k, v in keep.items()}, out / "confusion_matrices.png")
    ev.plot_tradeoff(agg, out / "accuracy_vs_compute.png")
    lr = keep["tfidf_logreg"][0]
    ev.plot_top_features(lr, out / "lr_top_features.png")

    # ---------------- explainability
    best_nn = keep.get("distilbert_mtl", keep["mtl_mlp"])[0]
    frame = ea["frame"]
    any_err = ~frame[[f"ok_{n}" for n in preds]].all(1)
    disagree = frame.loc[ea["disagreements"].index]
    real_err = frame[any_err & ~frame["likely_mislabeled"]]
    noisy = frame[frame["likely_mislabeled"]]
    rights = frame[~any_err]
    pick = pd.concat([disagree.head(3), real_err.head(4), noisy.head(2), rights.groupby("label").head(1)])
    pick = pick[~pick.index.duplicated()].head(args.n_explain)
    cases = []
    for _, row in pick.iterrows():
        t = row["text"]
        c_nn = int(best_nn.predict_proba([t])[0].argmax())
        exps = {
            f"LR exact → {LABELS[int(lr.predict_proba([t])[0].argmax())]}":
                xp.linear_contributions(lr, t, int(lr.predict_proba([t])[0].argmax())),
            f"{best_nn.name} occlusion → {LABELS[c_nn]}": xp.occlusion(best_nn.predict_proba, t, c_nn),
            f"{best_nn.name} LIME → {LABELS[c_nn]}": xp.lime_lite(best_nn.predict_proba, t, c_nn),
        }
        if hasattr(best_nn, "integrated_gradients"):
            exps[f"{best_nn.name} IG → {LABELS[c_nn]}"] = best_nn.integrated_gradients(t, c_nn)
        cases.append({
            "text": t, "gold": LABELS[row["label"]],
            "preds": {k: (LABELS[int(p.argmax())], float(p.max())) for k, p in
                      ((n, keep[n][0].predict_proba([t])[0]) for n in names)},
            "explanations": exps,
            "note": ("LIKELY MISLABELED (all models confidently disagree with gold) · " if row["likely_mislabeled"]
                     else "") + "tags: " + ", ".join(row["tags"]),
        })
    xp.render_report(cases, out / "explanations.html")

    sample = test0["text"].sample(min(150, len(test0)), random_state=0).tolist()
    faith = {}
    for nm, m in [("tfidf_logreg", lr), (best_nn.name, best_nn)]:
        faith[f"{nm} / occlusion"] = xp.faithfulness(m.predict_proba, sample,
                                                     lambda t, c, m=m: xp.occlusion(m.predict_proba, t, c))
        faith[f"{nm} / lime"] = xp.faithfulness(m.predict_proba, sample[:60],
                                                lambda t, c, m=m: xp.lime_lite(m.predict_proba, t, c, n_samples=200))
    faith["tfidf_logreg / exact-linear"] = xp.faithfulness(lr.predict_proba, sample,
                                                          lambda t, c: xp.linear_contributions(lr, t, c))

    # ---------------- outputs
    summary = {"data": source, "n": len(df), "seeds": args.seeds, "models": agg.to_dict("records"),
               "per_class_f1": per_class, "significance": sig, "top_confusions": ea["top_confusions"],
               "faithfulness": faith, "label_noise": ea["label_noise"],
               "mlp_history": {k: getattr(v[0], "history", None) for k, v in keep.items()}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    write_report(out, summary, agg, per_class, sig, ea, faith)
    log.info("done -> %s", out.resolve())
    print(agg.round(4).to_string(index=False))


def write_report(out, s, agg, per_class, sig, ea, faith):
    fmt = lambda x, d=3: f"{x:.{d}f}"  # noqa: E731
    L = [f"# Personal Health Mention classification — results\n",
         f"Data: **{s['data']}** · seeds: {s['seeds']} · split 70/15/15 stratified.\n"]
    if s["data"].startswith("synthetic"):
        L.append("> ⚠️ Synthetic, template-generated data: numbers demonstrate the pipeline, "
                 "not real-world performance. Run with `--data` on a real annotated corpus.\n")
    L += ["## Model comparison (test set, mean over seeds)\n",
          "| model | macro-F1 (±sd) | acc | AUC | aux fig-F1 | train s | ms / 1k posts | params | size MB |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in agg.iterrows():
        aux = "–" if pd.isna(r["aux_fig_f1"]) else fmt(r["aux_fig_f1"])
        L.append(f"| {r['model']} | {fmt(r['macro_f1'])} ± {fmt(r['macro_f1_std'])} | {fmt(r['accuracy'])} | "
                 f"{fmt(r['auc'])} | {aux} | {fmt(r['train_s'], 1)} | {fmt(r['infer_ms_per_1k'], 1)} | "
                 f"{int(r['n_params']):,} | {fmt(r['size_mb'], 2)} |")
    L += ["\n### Per-class F1\n", "| model | " + " | ".join(LABELS) + " |", "|---|---|---|---|"]
    L += [f"| {m} | " + " | ".join(fmt(v[l]) for l in LABELS) + " |" for m, v in per_class.items()]
    L += ["\n![confusions](confusion_matrices.png)\n", "![tradeoff](accuracy_vs_compute.png)\n",
          "## Significance (seed-0 test split)\n", "| comparison | McNemar p | A✓B✗ / A✗B✓ | Δ macro-F1 [95% CI] |",
          "|---|---|---|---|"]
    for k, v in sig.items():
        d = v["f1_diff"]
        L.append(f"| {k} | {v['p_value']:.3g} | {v['a_right_b_wrong']} / {v['a_wrong_b_right']} | "
                 f"{d['mean_diff']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] |")
    L += ["\n## Error analysis\n", "Accuracy by slice (seed-0 test split):\n", ea["slices"].to_markdown(index=False),
          "\nMost frequent confusions:\n"]
    for m, c in s["top_confusions"].items():
        L.append(f"- **{m}**: " + "; ".join(f"{k} ({v})" for k, v in c.items()))
    ln = ea["label_noise"]
    if ln:
        L.append(f"\n**Likely annotation errors:** {ln['flagged']} test posts where every model confidently "
                 f"(p ≥ 0.85) agrees on a label other than gold — {ln['share_of_errors']:.0%} of all errors. "
                 "These are re-annotation candidates rather than model failures.")
        if "precision" in ln:
            L.append(f"On synthetic data the injected label flips are known: detector precision "
                     f"{ln['precision']:.2f}, recall {ln['recall']:.2f}.")
    L += [f"\nModels disagreed on {len(ea['disagreements'])} test posts; full predictions in "
          "`test_predictions.csv`.\n",
          "## Explainability\n",
          "Per-post attributions are in **`explanations.html`** (LR exact contributions, occlusion, LIME-style "
          "surrogate; Integrated Gradients for DistilBERT). Global LR features: `lr_top_features.png`.\n",
          "Faithfulness (deleting the top-2 attributed words vs 2 random words; larger gap = more faithful):\n",
          "| model / method | n | Δp top-2 | Δp random-2 |", "|---|---|---|---|"]
    L += [f"| {k} | {v['n']} | {fmt(v['top_k_drop'])} | {fmt(v['random_k_drop'])} |" for k, v in faith.items()]
    (out / "report.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"total {time.time() - t0:.1f}s")
