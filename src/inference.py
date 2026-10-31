"""Classify new tickets and explain them. Shared by the dashboard (and usable from any script).

Two models (both from the notebooks):
- BGE sentence embeddings + Logistic Regression (notebook 02b, the best model): models/bge_logreg.joblib
- TF-IDF + Logistic Regression (notebook 02, the original model):             models/tfidf_logreg.joblib

Explanations are word-level SHAP values towards the predicted category:
- TF-IDF: the exact linear formula from notebook 03.
- BGE: Shapley values over words, where leaving a word out means deleting it and re-embedding the sentence
  (notebook 3b). Exact for short tickets; estimated with random word orders for long ones.
"""
import os
import re
from math import factorial

import joblib
import numpy as np
import pandas as pd

from . import config

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
MAX_EXACT_WORDS = 11        # 2^11 = 2,048 shortened sentences; above this SHAP is estimated
MAX_WORDS = 60              # longer tickets are explained on their first 60 words


def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def load_encoder(device=None):
    """The BGE sentence encoder (None if sentence-transformers isn't installed)."""
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        return SentenceTransformer(config.EMBEDDING_MODEL, device=device)
    except Exception:
        return None


class TicketClassifier:
    def __init__(self, encoder=None):
        self.tfidf_model = joblib.load(config.MODELS / "tfidf_logreg.joblib")
        self.tfidf = self.tfidf_model.named_steps["tfidfvectorizer"]
        self.tfidf_clf = self.tfidf_model.named_steps["logisticregression"]
        self.terms = self.tfidf.get_feature_names_out()
        self.background = np.load(config.MODELS / "shap_background.npy")
        self.classes = list(self.tfidf_clf.classes_)

        self.encoder = encoder
        self.bge_clf = None
        bundle_path = config.MODELS / "bge_logreg.joblib"
        if encoder is not None and bundle_path.exists():
            self.bge_clf = joblib.load(bundle_path)["classifier"]
            assert list(self.bge_clf.classes_) == self.classes
        self._cache = {}

        # the 96 known descriptions, with their theme and topic, for "similar tickets"
        t = pd.read_csv(config.PROCESSED / "tickets_clean.csv", usecols=["ticket_id", config.TEXT, config.TARGET])
        msgs = t.groupby([config.TEXT, config.TARGET]).size().rename("tickets").reset_index()
        cl = config.PROCESSED / "clusters_bge.csv"
        if cl.exists():
            c = pd.read_csv(cl).merge(t[["ticket_id", config.TEXT]], on="ticket_id").drop_duplicates(config.TEXT)
            msgs = msgs.merge(c[[config.TEXT, "theme", "topic", "topic_words"]], on=config.TEXT, how="left")
        self.messages = msgs
        self.msg_tfidf = self.tfidf.transform(msgs[config.TEXT])
        self.msg_emb = self.embed(msgs[config.TEXT].tolist()) if self.has_bge else None

    @property
    def has_bge(self):
        return self.bge_clf is not None

    # ------------------------------------------------------------------ BGE scores with a cache
    def embed(self, texts):
        return self.encoder.encode(list(texts), batch_size=128, normalize_embeddings=True, show_progress_bar=False)

    def bge_scores(self, texts):
        """Logits (texts x classes) of the BGE model; each distinct text is embedded once."""
        new = [s for s in dict.fromkeys(texts) if s not in self._cache]
        if new:
            self._cache.update(zip(new, self.bge_clf.decision_function(self.embed(new))))
            if len(self._cache) > 200_000:
                self._cache.clear()
        return np.array([self._cache[s] for s in texts])

    # ------------------------------------------------------------------ predictions
    def predict(self, text, model="bge"):
        """Probabilities for every class as a pandas Series (sorted, highest first)."""
        if model == "bge":
            p = softmax(self.bge_scores([text])[0])
        else:
            p = self.tfidf_model.predict_proba([text])[0]
        return pd.Series(p, index=self.classes).sort_values(ascending=False)

    def predict_many(self, texts, model="bge"):
        texts = [str(x) for x in texts]
        if model == "bge":
            p = softmax(self.bge_clf.decision_function(self.embed(texts)))
        else:
            p = self.tfidf_model.predict_proba(texts)
        return pd.DataFrame(p, columns=self.classes)

    @staticmethod
    def decision(label, confidence):
        review = confidence < config.CONFIDENCE_THRESHOLD
        return {"review": review, "queue": config.REVIEW_QUEUE if review else config.ROUTING[label]}

    # ------------------------------------------------------------------ explanations
    def explain_tfidf(self, text, label):
        x = self.tfidf.transform([text]).toarray()[0]
        contrib = (x - self.background) * self.tfidf_clf.coef_[self.classes.index(label)]
        present = np.where(x > 0)[0]
        order = present[np.argsort(-np.abs(contrib[present]))]
        return pd.DataFrame({"word": self.terms[order], "shap": contrib[order]}), "exact"

    def explain_bge(self, text, label, n_permutations=40, seed=config.SEED):
        """Word-level Shapley values towards `label`. Returns (DataFrame word/position/shap, method)."""
        words = text.split()[:MAX_WORDS]
        n, c = len(words), self.classes.index(label)
        if n == 0:
            return pd.DataFrame(columns=["word", "position", "shap"]), "exact"
        sentence = lambda keep: " ".join(w for j, w in enumerate(words) if keep[j])

        if n <= MAX_EXACT_WORDS:
            masks = np.arange(2 ** n)
            V = self.bge_scores([sentence([(m >> j) & 1 for j in range(n)]) for m in masks])[:, c]
            size = np.array([bin(m).count("1") for m in masks])
            weight = np.array([factorial(s) * factorial(n - s - 1) / factorial(n) for s in range(n)])
            phi = np.array([(weight[size[S]] * (V[S | (1 << j)] - V[S])).sum()
                            for j in range(n) for S in [masks[(masks >> j) & 1 == 0]]])
            method = "exact"
        else:
            rng = np.random.default_rng(seed)
            perms = []
            for _ in range(n_permutations // 2):            # antithetic pairs: an order and its reverse
                p = rng.permutation(n)
                perms += [p, p[::-1]]
            texts = []
            for p in perms:
                keep = np.zeros(n, bool)
                texts.append(sentence(keep))
                for j in p:
                    keep[j] = True
                    texts.append(sentence(keep))
            V = self.bge_scores(texts)[:, c].reshape(len(perms), n + 1)
            phi = np.zeros(n)
            for row, p in zip(V, perms):
                phi[p] += np.diff(row)
            phi /= len(perms)
            method = f"estimated ({len(perms)} random word orders)"
        return pd.DataFrame({"word": words, "position": range(n), "shap": phi}), method

    # ------------------------------------------------------------------ similar known descriptions
    def similar(self, text, k=5):
        if self.has_bge:
            sims = self.msg_emb @ self.embed([text])[0]
        else:
            q = self.tfidf.transform([text])
            sims = (self.msg_tfidf @ q.T).toarray().ravel()
        top = np.argsort(-sims)[:k]
        return self.messages.iloc[top].assign(similarity=sims[top]).reset_index(drop=True)


def highlight_html(shap_df, label_color="#4F46E5"):
    """The ticket text with each word shaded by its SHAP value (blue = towards the prediction, red = against)."""
    if shap_df.empty:
        return ""
    vmax = max(shap_df["shap"].abs().max(), 1e-9)
    spans = []
    for w, v in zip(shap_df["word"], shap_df["shap"]):
        a = min(abs(v) / vmax, 1) * 0.85
        rgb = "79,70,229" if v > 0 else "239,68,68"
        spans.append(f'<span title="SHAP {v:+.3f}" style="background: rgba({rgb},{a:.2f}); padding: 2px 5px; '
                     f'margin: 2px; border-radius: 6px; display: inline-block; color: {"#fff" if a > 0.5 else "#111827"}">'
                     f'{re.sub("[<>]", "", w)}</span>')
    return " ".join(spans)
