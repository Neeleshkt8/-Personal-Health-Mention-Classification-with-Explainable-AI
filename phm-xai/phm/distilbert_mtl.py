"""DistilBERT multi-task classifier (PyTorch + Hugging Face Transformers).

Architecture
    DistilBERT encoder -> [CLS] hidden state (768)
        -> main head:  Dropout -> Linear(768, 3)   (LITERAL_PHM / LITERAL_OTHER / FIGURATIVE)
        -> aux head:   Dropout -> Linear(768, 1)   (figurative-language detection)
    loss = CrossEntropy(main) + lambda_aux * BCEWithLogits(aux)

Explainability: Integrated Gradients on the input word embeddings, merged from
WordPiece sub-tokens back to whitespace words so attributions align with the
model-agnostic explainers in explain.py.

Requires: pip install torch transformers  (GPU recommended; CPU works for small data)
"""
from __future__ import annotations

import io
import time

import numpy as np

try:
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    TORCH_OK = True
except ImportError:  # keep the module importable so the rest of the pipeline runs
    TORCH_OK = False

from sklearn.metrics import f1_score

from .text import clean

if TORCH_OK:

    class _Net(nn.Module):
        def __init__(self, name: str, n_main: int = 3, dropout: float = 0.1):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(name)
            h = self.encoder.config.hidden_size
            self.main_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(h, n_main))
            self.aux_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(h, 1))

        def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds)
            cls = out.last_hidden_state[:, 0]
            return self.main_head(cls), self.aux_head(cls).squeeze(-1)


class DistilBertMTL:
    def __init__(self, model_name="distilbert-base-uncased", lambda_aux=0.5, epochs=3, batch=16, lr=3e-5,
                 max_len=128, warmup=0.1, patience=1, seed=42, device=None, name=None):
        if not TORCH_OK:
            raise ImportError("DistilBertMTL needs `pip install torch transformers`")
        self.model_name, self.lam, self.epochs, self.batch, self.lr = model_name, lambda_aux, epochs, batch, lr
        self.max_len, self.warmup, self.patience, self.seed = max_len, warmup, patience, seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.name = name or ("distilbert_mtl" if lambda_aux > 0 else "distilbert_stl")

    def _enc(self, texts):
        return self.tok([clean(t) for t in texts], truncation=True, max_length=self.max_len,
                        padding=True, return_tensors="pt")

    def fit(self, train, val):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.net = _Net(self.model_name).to(self.device)
        texts = train["text"].tolist()
        y = torch.tensor(train["label"].to_numpy())
        ya = torch.tensor(train["figurative"].to_numpy(), dtype=torch.float)
        n_steps = self.epochs * ((len(texts) + self.batch - 1) // self.batch)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, int(self.warmup * n_steps), n_steps)
        ce, bce = nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()
        best, best_state, bad = -1.0, None, 0
        self.history = []
        g = torch.Generator().manual_seed(self.seed)
        for ep in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(texts), generator=g)
            for i in range(0, len(texts), self.batch):
                idx = perm[i : i + self.batch]
                enc = self._enc([texts[j] for j in idx.tolist()]).to(self.device)
                main, aux = self.net(enc["input_ids"], enc["attention_mask"])
                loss = ce(main, y[idx].to(self.device)) + self.lam * bce(aux, ya[idx].to(self.device))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                sched.step()
            f1 = f1_score(val["label"], self.predict_proba(val["text"]).argmax(1), average="macro")
            self.history.append({"epoch": ep + 1, "val_macro_f1": round(float(f1), 4)})
            if f1 > best:
                best, bad = f1, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad > self.patience:
                    break
        self.net.load_state_dict(best_state)
        return self

    @torch.no_grad() if TORCH_OK else (lambda f: f)
    def _predict(self, texts, bs=64):
        self.net.eval()
        texts = list(texts)
        P, Q = [], []
        for i in range(0, len(texts), bs):
            enc = self._enc(texts[i : i + bs]).to(self.device)
            main, aux = self.net(enc["input_ids"], enc["attention_mask"])
            P.append(torch.softmax(main, -1).cpu().numpy())
            Q.append(torch.sigmoid(aux).cpu().numpy())
        return np.concatenate(P), np.concatenate(Q)

    def predict_proba(self, texts):
        return self._predict(texts)[0]

    def predict_aux(self, texts):
        return self._predict(texts)[1]

    def integrated_gradients(self, text: str, target: int | None = None, steps: int = 32):
        """Return [(word, attribution)] using IG on word embeddings (zero baseline)."""
        self.net.eval()
        enc = self._enc([text]).to(self.device)
        ids, am = enc["input_ids"], enc["attention_mask"]
        emb = self.net.encoder.embeddings.word_embeddings(ids).detach()
        if target is None:
            target = int(self.predict_proba([text])[0].argmax())
        total = torch.zeros_like(emb)
        for a in torch.linspace(1 / steps, 1, steps):
            x = (a * emb).requires_grad_(True)
            logits, _ = self.net(attention_mask=am, inputs_embeds=x)
            torch.softmax(logits, -1)[0, target].backward()
            total += x.grad
            self.net.zero_grad(set_to_none=True)
        attr = ((emb * total / steps).sum(-1))[0].detach().cpu().numpy()
        toks = self.tok.convert_ids_to_tokens(ids[0])
        words, scores = [], []
        for t, s in zip(toks, attr):
            if t in ("[CLS]", "[SEP]", "[PAD]"):
                continue
            if t.startswith("##") and words:
                words[-1] += t[2:]
                scores[-1] += s
            else:
                words.append(t)
                scores.append(s)
        return list(zip(words, map(float, scores)))

    @property
    def n_params(self):
        return int(sum(p.numel() for p in self.net.parameters()))

    @property
    def size_bytes(self):
        buf = io.BytesIO()
        torch.save(self.net.state_dict(), buf)
        return buf.tell()
