# Personal Health Mention Classification with Explainable AI

Classifies social-media posts by **how a health term is used**:

| label | meaning | example |
|---|---|---|
| `LITERAL_PHM` | a real personal health report | "woke up with a migraine again, calling my gp" |
| `LITERAL_OTHER` | literal, but not a personal report (news, info, awareness) | "new study links poor sleep to migraine" |
| `FIGURATIVE` | health word used figuratively or hyperbolically | "this group project is giving me a migraine" |

**Auxiliary task:** does the post contain figurative language at all? This is always true for `FIGURATIVE`, but also true for literal reports that use a metaphor ("my migraine feels like a jackhammer"), so it is a related but distinct signal for multi-task learning.

## Pipeline

```
data (CSV or synthetic) → clean → stratified 70/15/15 split, repeated over seeds
   ├─ TF-IDF (word 1-2g + char 2-5g) + Logistic Regression   (C tuned by CV)
   ├─ Single-task MLP   (ablation: λ_aux = 0)
   ├─ Multi-task MLP    shared layer → main softmax(3) + aux sigmoid   (numpy, runs anywhere)
   └─ DistilBERT multi-task  [CLS] → main head + aux head, loss = CE + λ·BCE   (--distilbert)
        ↓
evaluation: macro-F1 / acc / AUC / per-class F1 / aux-F1, mean ± sd over seeds
            McNemar test + bootstrap CI on paired predictions
            train time, inference ms per 1k posts, parameters, model size
        ↓
error analysis: confusion pairs, accuracy by slice (simile, near-miss phrase, 3rd person,
                slang/emoji, length…), model disagreements, likely-mislabel detection
        ↓
explainability: LR exact contributions · occlusion · LIME-style surrogate · Integrated Gradients (BERT)
                + faithfulness test (top-k deletion vs random deletion)
        ↓
reports/: report.md · explanations.html · confusion_matrices.png · accuracy_vs_compute.png
          lr_top_features.png · test_predictions.csv · error_slices.csv · summary.json
```

## Run

```bash
pip install -r requirements.txt
python run_pipeline.py                                       # synthetic data, ~20 s on CPU
python run_pipeline.py --data posts.csv                      # your labelled data
python run_pipeline.py --data posts.csv --distilbert --bert-ablation --seeds 0   # + DistilBERT (GPU)
pytest -q
```

**Google Colab (for DistilBERT):** upload the folder, select a GPU runtime, then run
`!pip install -r requirements.txt` and `!python run_pipeline.py --data posts.csv --distilbert --bert-ablation --seeds 0`.

**CSV format:** a `text` column, plus a `label` column holding ids 0–2 or names (`personal`, `non-personal`, `figurative`, …; see `_ALIASES` in `phm/data.py`). An optional `figurative` 0/1 column gives the auxiliary label. If it's missing, it is derived from the label plus simile/metaphor markers. Public health-mention corpora from Twitter and Reddit exist in the research literature; check each dataset's license and terms before use.

## Results on the bundled synthetic data (3 seeds)

| model | macro-F1 | aux fig-F1 | ms / 1k posts | params |
|---|---|---|---|---|
| TF-IDF + LogReg | 0.952 ± 0.007 | – | ~85 | 17k |
| Single-task MLP | 0.955 ± 0.008 | – | ~18 | 255k |
| Multi-task MLP | 0.955 ± 0.006 | 0.929 | ~18 | 255k |

- No model is significantly better on this data (McNemar p = 1). The auxiliary task does not raise main-task F1 here, but it gives you a working figurative-language detector for free.
- About 47% of test errors are flagged as likely annotation errors. On synthetic data the injected label flips are known, so the detector's precision can be measured: 1.00, with recall 0.75.
- The explanations are faithful: deleting the top-2 attributed words lowers the predicted-class probability by about 0.22–0.27, versus about 0.07 for 2 random words.
- LR inference is slower than the MLP because the char n-gram vectoriser dominates its cost.

> ⚠️ The synthetic data is template-generated (with truncation, typos and 3% label noise for realism). These numbers show that the pipeline works; they are **not** real-world performance. On real posts, expect lower scores and a larger gap between DistilBERT and TF-IDF.

## Layout
```
phm/data.py            synthetic generator, CSV loader, split
phm/text.py            social-media normalisation (URLs, @users, hashtags, elongation)
phm/models.py          TfidfLogReg, MultiTaskMLP (numpy, gradient-checked), timing helpers
phm/distilbert_mtl.py  DistilBERT multi-task model + Integrated Gradients
phm/explain.py         linear contributions, occlusion, LIME-lite, faithfulness, HTML report
phm/evaluate.py        metrics, McNemar, bootstrap, error slices, label-noise detector, plots
run_pipeline.py        end-to-end runner and report writer
tests/                 12 tests incl. numerical gradient check and an end-to-end run
```
