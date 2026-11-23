"""Classical + lightweight neural models sharing one interface:

    fit(train_df, val_df) -> self
    predict_proba(texts) -> (n, 3) array
    predict_aux(texts)   -> (n,) figurative-language probability, or None
    n_params, size_bytes, name
"""
from __future__ import annotations

import io
import pickle
import time

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .text import clean


def _size(obj) -> int:
    buf = io.BytesIO()
    pickle.dump(obj, buf)
    return buf.tell()


class TfidfLogReg:
    """Word 1-2gram + char 2-5gram TF-IDF -> multinomial logistic regression (C tuned by CV)."""

    name = "tfidf_logreg"

    def fit(self, train, val=None, seed: int = 42):
        self.word = TfidfVectorizer(preprocessor=clean, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.char = TfidfVectorizer(preprocessor=clean, analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                    sublinear_tf=True, max_features=30000)
        X = self._fit_features(train["text"])
        grid = GridSearchCV(
            LogisticRegression(max_iter=3000, class_weight="balanced"),
            {"C": [0.5, 2, 8]}, scoring="f1_macro", cv=StratifiedKFold(3, shuffle=True, random_state=seed), n_jobs=1,
        )
        grid.fit(X, train["label"])
        self.clf, self.best_C = grid.best_estimator_, grid.best_params_["C"]
        return self

    def _fit_features(self, texts):
        return sparse.hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr()

    def features(self, texts):
        return sparse.hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()

    def feature_names(self):
        return np.concatenate([self.word.get_feature_names_out(),
                               ["c:" + f for f in self.char.get_feature_names_out()]])

    def predict_proba(self, texts):
        return self.clf.predict_proba(self.features(list(texts)))

    def predict_aux(self, texts):
        return None

    @property
    def n_params(self):
        return int(self.clf.coef_.size + self.clf.intercept_.size)

    @property
    def size_bytes(self):
        return _size((self.word, self.char, self.clf))


class MultiTaskMLP:
    """Shared-encoder multi-task network in pure numpy (offline stand-in for DistilBERT-MTL).

    TF-IDF -> Dense(h, ReLU, dropout) -> [main head: softmax(3)] + [aux head: sigmoid(1)]
    loss = CE(main) + lambda_aux * BCE(figurative) + l2
    Trained with Adam + early stopping on validation macro-F1.
    Setting lambda_aux=0 gives the single-task ablation.
    """

    def __init__(self, hidden=128, lambda_aux=0.5, lr=2e-3, epochs=30, batch=64, dropout=0.3,
                 l2=1e-5, patience=4, seed=42, name=None):
        self.h, self.lam, self.lr, self.epochs, self.batch = hidden, lambda_aux, lr, epochs, batch
        self.dropout, self.l2, self.patience, self.seed = dropout, l2, patience, seed
        self.name = name or ("mtl_mlp" if lambda_aux > 0 else "stl_mlp")

    # ---- forward ----------------------------------------------------------
    def _forward(self, X, train=False, rng=None):
        W1, b1, W2, b2, w3, b3 = self.params
        H = np.maximum(0, X @ W1 + b1)
        mask = None
        if train and self.dropout:
            mask = (rng.random(H.shape) > self.dropout) / (1 - self.dropout)
            H = H * mask
        Z = H @ W2 + b2
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(1, keepdims=True)
        q = 1 / (1 + np.exp(-(H @ w3 + b3)))
        return H, mask, P, q.ravel()

    def _loss_and_grads(self, xb, Yb, yab, rng=None):
        """Joint loss CE + lambda*BCE (+L2) and its analytic gradients (dropout if rng given)."""
        n = xb.shape[0]
        H, mask, P, q = self._forward(xb, rng is not None, rng)
        W1, b1, W2, b2, w3, b3 = self.params
        eps = 1e-7
        loss = (-np.log(P[np.arange(n), Yb.argmax(1)] + eps).mean()
                - self.lam * (yab * np.log(q + eps) + (1 - yab) * np.log(1 - q + eps)).mean()
                + 0.5 * self.l2 * sum(float((W * W).sum()) for W in (W1, W2, w3)))
        dZ = (P - Yb) / n
        da = (self.lam * (q - yab) / n)[:, None]
        dH = (dZ @ W2.T + da @ w3.T) * (H > 0)
        if mask is not None:
            dH *= mask
        grads = [xb.T @ dH + self.l2 * W1, dH.sum(0), H.T @ dZ + self.l2 * W2, dZ.sum(0),
                 H.T @ da + self.l2 * w3, da.sum(0)]
        return float(loss), grads

    def fit(self, train, val, seed=None):
        rng = np.random.default_rng(self.seed if seed is None else seed)
        self.vec = TfidfVectorizer(preprocessor=clean, ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                   max_features=20000)
        X = self.vec.fit_transform(train["text"]).astype(np.float32).tocsr()
        Xv = self.vec.transform(val["text"]).astype(np.float32).tocsr()
        y, ya = train["label"].to_numpy(), train["figurative"].to_numpy().astype(np.float32)
        Y = np.eye(3, dtype=np.float32)[y]
        d = X.shape[1]
        self.params = [
            rng.normal(0, np.sqrt(2 / d), (d, self.h)).astype(np.float32), np.zeros(self.h, np.float32),
            rng.normal(0, np.sqrt(2 / self.h), (self.h, 3)).astype(np.float32), np.zeros(3, np.float32),
            rng.normal(0, np.sqrt(2 / self.h), (self.h, 1)).astype(np.float32), np.zeros(1, np.float32),
        ]
        m = [np.zeros_like(p) for p in self.params]
        v = [np.zeros_like(p) for p in self.params]
        step, best, best_params, bad = 0, -1, None, 0
        self.history = []
        for ep in range(self.epochs):
            order = rng.permutation(X.shape[0])
            for i in range(0, len(order), self.batch):
                idx = order[i : i + self.batch]
                _, grads = self._loss_and_grads(X[idx], Y[idx], ya[idx], rng)
                step += 1
                for k, g in enumerate(grads):  # Adam
                    g = np.asarray(g, dtype=np.float32)
                    m[k] = 0.9 * m[k] + 0.1 * g
                    v[k] = 0.999 * v[k] + 0.001 * g * g
                    self.params[k] -= self.lr * (m[k] / (1 - 0.9 ** step)) / (np.sqrt(v[k] / (1 - 0.999 ** step)) + 1e-8)
            f1 = f1_score(val["label"], self._forward(Xv)[2].argmax(1), average="macro")
            self.history.append({"epoch": ep + 1, "val_macro_f1": round(float(f1), 4)})
            if f1 > best + 1e-4:
                best, best_params, bad = f1, [p.copy() for p in self.params], 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.params = best_params
        return self

    def predict_proba(self, texts):
        return self._forward(self.vec.transform(list(texts)).astype(np.float32))[2]

    def predict_aux(self, texts):
        if self.lam == 0:  # aux head is untrained in the single-task ablation
            return None
        return self._forward(self.vec.transform(list(texts)).astype(np.float32))[3]

    @property
    def n_params(self):
        return int(sum(p.size for p in self.params))

    @property
    def size_bytes(self):
        return _size((self.vec, self.params))


def timed_fit(model, train, val):
    t0 = time.perf_counter()
    model.fit(train, val)
    return time.perf_counter() - t0


def inference_ms_per_1k(model, texts, repeats=3):
    texts = list(texts)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.predict_proba(texts)
        best = min(best, time.perf_counter() - t0)
    return 1000 * best * 1000 / max(1, len(texts))
