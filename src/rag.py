"""GenAI assistant using RAG (Retrieval-Augmented Generation).

How it works:
1. Knowledge base: short factual documents generated from every saved result (category stats, both classifiers,
   SHAP for both models, recurring themes and topics, anomalies, data profile), plus the written interpretations
   from every notebook, the evaluation report and the README, one document per section.
2. Retrieval: hybrid search. TF-IDF keyword similarity is combined with BGE sentence-embedding similarity
   (semantic search) when sentence-transformers is installed; otherwise keyword search alone is used.
3. Generation: an LLM (Mistral, Claude or any OpenAI-compatible model) writes the answer using ONLY the retrieved
   documents. Without an API key, an offline mode answers with the most relevant document text instead.
4. Grounding check: every number in the answer must appear in the retrieved documents.

Used by notebook 06 and the Streamlit dashboard.
"""
import json
import os
import re
import time

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config

SYSTEM_PROMPT = (
    "You are the analytics assistant for an IT support ticket project. Answer using ONLY the context documents. "
    "Quote numbers exactly as they appear in the context. Never invent, estimate or calculate new numbers "
    "(no differences, ratios or rounding). If the context does not contain the answer, say so plainly. "
    "The project has two classifiers: TF-IDF + Logistic Regression (the original model) and BGE sentence embeddings + "
    "Logistic Regression (the best model). When a number describes a classifier, say which one; clustering, SHAP, "
    "anomaly and data results are separate analyses, not classifiers. "
    "Be concise: a short paragraph or a few bullet points. Cite the document ids you used in square brackets, "
    "e.g. [category:bug]. You assist people; never suggest closing or changing tickets automatically."
)


# ------------------------------------------------------------------ environment (.env)
def load_env(path=config.ENV_FILE):
    """Read KEY=value lines from .env into the environment (existing variables win).
    A line without '=' is taken as a Mistral API key, so a file holding only the key also works."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            name, value = line.split("=", 1)
            os.environ.setdefault(name.strip().removeprefix("export "), value.strip().strip("\"'"))
        else:
            os.environ.setdefault("MISTRAL_API_KEY", line.strip("\"'"))


load_env()


# ------------------------------------------------------------------ helpers
def mask(text):
    """Remove sensitive details before any text leaves the machine."""
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "<EMAIL>", str(text))
    text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<IP>", text)
    text = re.sub(r"\+?\d[\d\s-]{8,}\d", "<PHONE>", text)
    return text


def pct(x):
    return f"{100 * x:.1f}%"


def _numbers(text):
    return {round(float(n.replace(",", "")), 1) for n in re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?", text)}


def numbers_grounded(answer, context):
    """True if every number in the answer also appears in the context (small counting numbers are allowed)."""
    allowed = _numbers(context) | {float(i) for i in range(11)}
    return all(n in allowed for n in _numbers(answer))


def _read(p, name, **kw):
    f = p / name
    if not f.exists():
        return None
    return json.load(open(f)) if name.endswith(".json") else pd.read_csv(f, **kw)


# ------------------------------------------------------------------ knowledge base
def build_knowledge_base(processed=config.PROCESSED):
    p = processed
    t = pd.read_csv(p / "tickets_clean.csv", parse_dates=["created_at"])
    pred = pd.read_csv(p / "predictions.csv")
    report = pd.read_csv(p / "classification_report.csv", index_col=0)
    metrics = json.load(open(p / "metrics.json"))
    comparison = pd.read_csv(p / "model_comparison.csv", index_col=0)
    cv = pd.read_csv(p / "cv_scores.csv", index_col=0)
    shap_g = pd.read_csv(p / "shap_global.csv")
    themes = pd.read_csv(p / "cluster_summary.csv")
    volume = pd.read_csv(p / "volume_anomalies.csv")
    hot = pd.read_csv(p / "hotspots.csv")
    odd = pd.read_csv(p / "ticket_anomalies.csv")
    validation = pd.read_csv(p / "anomaly_validation.csv")

    # results of the embedding notebooks (02b, 03b, 04b); optional so notebooks 01-06 still work on their own
    emb_metrics = _read(p, "embedding_metrics.json")
    emb_cv = _read(p, "embedding_cv_scores.csv")
    emb_report = _read(p, "embedding_class_report.csv", index_col=0)
    emb_rule = _read(p, "embedding_confidence_rule.csv")
    emb_changes = _read(p, "embedding_message_changes.csv")
    emb_per_class = _read(p, "embedding_per_class_f1.csv", index_col=0)
    shap_bge = _read(p, "shap_bge_global.csv")
    shap_checks = _read(p, "shap_bge_checks.json")
    shap_del = _read(p, "shap_bge_deletion_test.csv")
    shap_overall = _read(p, "shap_overall_comparison.csv")
    shap_examples = _read(p, "shap_bge_examples.csv")
    themes_bge = _read(p, "cluster_summary_bge.csv")
    topics = _read(p, "topic_clusters_bge.csv")
    clu_cmp = _read(p, "clustering_bge_vs_lsa.csv")
    clu_sim = _read(p, "clustering_similarity.csv", index_col=0)
    clu_scan = _read(p, "clustering_bge_scan.csv")
    has_bge = emb_metrics is not None

    t = t.merge(pred[["ticket_id", "needs_review", "confidence"]], on="ticket_id")
    docs = []

    def add(doc_id, source, title, text):
        docs.append({"doc_id": doc_id, "source": source, "title": title, "text": mask(text)})

    # ---------------------------------------------------------------- overview and operations
    shares = t[config.TARGET].value_counts(normalize=True)
    add("overview", "analytics", "Dataset overview and ticket volume",
        f"The analysis uses a stratified sample of {len(t)} tickets created between {t['created_at'].min().date()} "
        f"and {t['created_at'].max().date()}. There are {shares.size} categories and they are balanced: the largest is "
        f"{shares.index[0]} with {pct(shares.iloc[0])} of tickets and the smallest is {shares.index[-1]} with "
        f"{pct(shares.iloc[-1])}. Ticket volume is flat over time at about "
        f"{t.groupby(t['created_at'].dt.to_period('M')).size().mean():.0f} tickets per month. "
        f"{pct(t['status'].eq('resolved').mean())} of tickets are resolved and "
        f"{pct(t['status'].isin(['open', 'in_progress', 'on_hold']).mean())} are still open, in progress or on hold. "
        f"Overall {pct(t['customer_sentiment'].isin(['negative', 'very_negative']).mean())} of customers are negative. "
        f"The ticket messages are built from only {t[config.TEXT].nunique()} unique sentences.")

    profile = []
    for col in ["priority", "channel", "product_area", "customer_segment", "platform", "region", "sla_plan"]:
        s = t[col].value_counts(normalize=True)
        profile.append(f"{col.replace('_', ' ')}: " + ", ".join(f"{k} {pct(v)}" for k, v in s.items()))
    mix = pd.crosstab(t["product_area"], t[config.TARGET], normalize="index")
    add("analytics:profile", "analytics", "Ticket profile: priority, channel, product area, segment, platform, region, SLA",
        "Share of tickets by field. " + "; ".join(profile) + f". Attachments are included in "
        f"{pct(t['has_attachment'].mean())} of tickets. These fields do not tell the category apart: within every product "
        f"area each category's share lies between {pct(mix.values.min())} and {pct(mix.values.max())}, so the "
        "classifiers use only the ticket description.")

    sat = t.assign(neg=t["customer_sentiment"].isin(["negative", "very_negative"])).groupby(config.TARGET).agg(
        neg=("neg", "mean"), csat=("csat_score", "mean")).sort_values("neg", ascending=False)
    add("analytics:satisfaction", "analytics", "Customer satisfaction by category: which customers are unhappy",
        "Customer sentiment and satisfaction (CSAT) differ a lot by category. Share of negative or very negative "
        "(unhappy, frustrated) customers and average CSAT per category, from worst to best: " +
        "; ".join(f"{c}: {pct(r.neg)} negative, CSAT {r.csat:.2f}" for c, r in sat.iterrows()) +
        ". Problem categories make customers much unhappier than requests such as how_to and feature_request.")

    res_t = t[t["status"] == "resolved"]
    by_cat = res_t.groupby(config.TARGET)["resolution_time_hours"].median()
    by_pri = res_t.groupby("priority")["resolution_time_hours"].median().reindex(["urgent", "high", "medium", "low"])
    by_sla = res_t.groupby("sla_plan")["resolution_time_hours"].median()
    reopen_pri = t.groupby("priority")["reopened"].mean().reindex(["urgent", "high", "medium", "low"])
    add("analytics:resolution", "analytics", "Resolution time: how long tickets take to resolve, reopen rate",
        "Median resolution time of resolved tickets by priority: " +
        "; ".join(f"{k}: {v:.1f} hours" for k, v in by_pri.items()) +
        ". By SLA plan: " + "; ".join(f"{k}: {v:.1f} hours" for k, v in by_sla.items()) +
        ". By category the median resolution time is almost the same everywhere: " +
        "; ".join(f"{k}: {v:.1f} hours" for k, v in by_cat.items()) +
        f". Priority drives how fast tickets are resolved, the category does not. The reopen rate is "
        f"{pct(t['reopened'].mean())} overall; by priority: " +
        "; ".join(f"{k}: {pct(v)}" for k, v in reopen_pri.items()) + ".")

    queues = pred["queue"].value_counts()
    add("analytics:routing", "analytics", "Routing: tickets per support queue and manual review",
        "Predicted tickets per support queue with the TF-IDF model (low-confidence tickets go to Manual Triage): " +
        "; ".join(f"{q}: {n} tickets" for q, n in queues.items()) +
        ". The category-to-queue routing table is an assumption because the data has no assignment group: " +
        "; ".join(f"{c} goes to {q}" for c, q in config.ROUTING.items()) + ".")

    # ---------------------------------------------------------------- one document per category
    for cat, g in t.groupby(config.TARGET):
        words = ", ".join(shap_g[shap_g["category"] == cat].sort_values("mean_shap", ascending=False)["term"].head(6))
        cat_themes = themes[themes["category"] == cat].sort_values("tickets", ascending=False)
        theme_txt = "; ".join(f"{r.theme} ({r.tickets} tickets, e.g. \"{r.example}\")" for r in cat_themes.itertuples())
        r = report.loc[cat]
        text = (f"Category {cat} has {len(g)} tickets ({pct(len(g) / len(t))} of all tickets) and is routed to "
                f"{config.ROUTING[cat]}. Customer sentiment is negative for "
                f"{pct(g['customer_sentiment'].isin(['negative', 'very_negative']).mean())} of these tickets, average CSAT "
                f"(rated tickets) is {g['csat_score'].mean():.2f}, the reopen rate is {pct(g['reopened'].mean())} and the "
                f"median resolution time of resolved tickets is "
                f"{res_t.loc[res_t[config.TARGET] == cat, 'resolution_time_hours'].median():.1f} hours. "
                f"TF-IDF + Logistic Regression performance for {cat}: precision {r['precision']:.2f}, recall "
                f"{r['recall']:.2f}, F1 {r['f1-score']:.2f}, ROC-AUC {r['roc_auc']:.2f}; {pct(g['needs_review'].mean())} "
                f"of its tickets fall below the confidence threshold and go to manual review. ")
        if has_bge:
            b = emb_report.loc[cat]
            bw = ", ".join(shap_bge[shap_bge["category"] == cat].sort_values("mean_shap", ascending=False)["term"].head(6))
            text += (f"BGE embeddings + Logistic Regression (best model) performance for {cat}: precision "
                     f"{b['precision']:.2f}, recall {b['recall']:.2f}, F1 {b['f1-score']:.2f}, ROC-AUC {b['roc_auc']:.2f}. ")
        text += f"SHAP words that push tickets into {cat} with the TF-IDF model: {words}. "
        if has_bge:
            text += f"SHAP words with the BGE model: {bw}. "
        text += f"Recurring themes in {cat} (notebook 04, TF-IDF + LSA clustering): {theme_txt}."
        if has_bge:
            bt = themes_bge[themes_bge["category"] == cat].sort_values("tickets", ascending=False)
            text += " Themes with BGE clustering (notebook 4b): " + "; ".join(
                f"{x.theme} ({x.tickets} tickets, e.g. \"{x.example}\")" for x in bt.itertuples()) + "."
        add(f"category:{cat}", "category",
            f"Category {cat}: volume, satisfaction, resolution, model performance, SHAP words and recurring themes", text)

        shap_txt = (f"According to SHAP, the words that most push the TF-IDF model to predict {cat} are: {words}. "
                    f"These are averaged SHAP values over all {cat} tickets.")
        if has_bge:
            shap_txt += (f" For the BGE embedding model, exact word-level SHAP gives these top words for {cat}: " +
                         ", ".join(f"{x.term} ({x.mean_shap:.2f})" for x in shap_bge[shap_bge["category"] == cat]
                                   .sort_values("mean_shap", ascending=False).head(8).itertuples()) + ".")
        add(f"shap:{cat}", "shap", f"Which words drive {cat} predictions (SHAP explainability)", shap_txt)
        add(f"themes:{cat}", "theme", f"Recurring issues in {cat} (clustering)",
            f"Clustering (notebook 04, TF-IDF + LSA) found {len(cat_themes)} recurring issue themes in {cat}: {theme_txt}." +
            ("" if not has_bge else " With BGE embeddings (notebook 4b) the themes are: " + "; ".join(
                f"{x.theme} ({x.tickets} tickets, {x.unique_sentences} descriptions, e.g. \"{x.example}\")"
                for x in themes_bge[themes_bge["category"] == cat].sort_values("tickets", ascending=False).itertuples()) + "."))

    # ---------------------------------------------------------------- TF-IDF model
    add("model:summary", "model", "How good is the TF-IDF classifier: F1, precision, recall, accuracy",
        f"The original classifier is {metrics['model']}, evaluated with {metrics['evaluation']}. Macro F1 is "
        f"{metrics['macro_f1']:.3f}, weighted F1 {metrics['weighted_f1']:.3f}, macro precision "
        f"{metrics['macro_precision']:.3f}, macro recall {metrics['macro_recall']:.3f}, accuracy {metrics['accuracy']:.3f} "
        f"and macro ROC-AUC {metrics['roc_auc_macro']:.3f}. Tickets with confidence below "
        f"{metrics['confidence_threshold']:.0%} go to manual review; that is {pct(metrics['review_share'])} of tickets. "
        f"The weakest categories are {', '.join(report.loc[report.index.isin(config.ROUTING)].nsmallest(2, 'f1-score').index)}, "
        f"which have only 3 different sentences each, so the model rarely sees new wording for them." +
        ("" if not has_bge else f" The BGE embedding model is better: macro F1 {emb_metrics['f1_macro']:.3f}."))
    best = cv["f1_macro_mean"].drop("Majority baseline")
    add("model:comparison", "model", "TF-IDF model comparison: Logistic Regression vs SVM vs Naive Bayes",
        "With TF-IDF features, on the held-out test messages the models scored: " +
        "; ".join(f"{m}: macro F1 {r.f1_macro:.3f}, ROC-AUC {r.roc_auc_macro:.3f}" for m, r in comparison.iterrows()) +
        ". In grouped 5-fold cross-validation the macro F1 was: " +
        "; ".join(f"{m}: {v:.3f}" for m, v in cv["f1_macro_mean"].items()) +
        f". Naive Bayes, Linear SVM and Logistic Regression tie on F1 ({best.max():.3f}) because the test set has so few "
        "distinct sentences; Logistic Regression was chosen because it gives probabilities for the confidence score and "
        "its SHAP values can be computed exactly.")
    add("model:leakage", "model", "Data leakage and train/test split",
        "The dataset has only 96 unique ticket messages repeated across all tickets, and each message always has the same "
        "category. A random train/test split therefore puts the same sentences in both sets and gives a misleading macro F1 "
        "of 1.0. The project splits by message instead (grouped split), so the test only contains unseen wording. The "
        "columns agent_first_reply and resolution_summary reveal the category and are never used as model inputs, and "
        "outcome columns such as status, sentiment and CSAT are only known after a ticket is handled.")
    add("model:shap", "model", "How SHAP explanations work for the TF-IDF model",
        "SHAP gives each word in a ticket a score for how much it pushed the prediction towards or away from a category. "
        "For the linear TF-IDF + Logistic Regression model the SHAP value of a word is its model weight times the "
        "difference between its TF-IDF value in the ticket and its average TF-IDF value. The values add up exactly to the "
        "model score. Some filler words such as 'my' and 'the' carry weight because the synthetic sentences are built "
        "from templates.")

    # ---------------------------------------------------------------- embedding models (02b)
    if has_bge:
        rep_best = emb_cv.loc[emb_cv.groupby("representation")["f1_macro_mean"].idxmax()].sort_values(
            "f1_macro_mean", ascending=False)
        add("model:embeddings", "model",
            "Best model: sentence embeddings (BGE, MPNet, MiniLM) vs TF-IDF, which classifier is best",
            "Notebook 2b replaced TF-IDF with pre-trained sentence embeddings that capture meaning: MiniLM "
            "(all-MiniLM-L6-v2), MPNet (all-mpnet-base-v2) and BGE (BAAI/bge-base-en-v1.5). Each was paired with "
            "Logistic Regression, Linear SVM, k-nearest neighbours and Random Forest, using the same message-level split "
            "and grouped 5-fold cross-validation as the TF-IDF models. Best macro F1 per representation in grouped CV: " +
            "; ".join(f"{r.representation} with {r.classifier}: {r.f1_macro_mean:.3f} (std {r.f1_macro_std:.3f})"
                      for r in rep_best.itertuples()) +
            f". The chosen model is {emb_metrics['model']}: out-of-fold macro F1 {emb_metrics['f1_macro']:.3f}, accuracy "
            f"{emb_metrics['accuracy']:.3f}, macro precision {emb_metrics['precision_macro']:.3f}, macro recall "
            f"{emb_metrics['recall_macro']:.3f}, weighted F1 {emb_metrics['f1_weighted']:.3f} and macro ROC-AUC "
            f"{emb_metrics['roc_auc_macro']:.3f}, compared with macro F1 {metrics['macro_f1']:.3f} and accuracy "
            f"{metrics['accuracy']:.3f} for TF-IDF + Logistic Regression. Linear classifiers work best on embeddings; "
            "Random Forest is weakest because meaning is spread over hundreds of dimensions and there are only about 77 "
            "training sentences per fold.")
        add("model:embedding_grid", "model", "All embedding x classifier scores (grouped cross-validation table)",
            "Grouped 5-fold cross-validation macro F1 (mean and std over folds) and out-of-fold ROC-AUC for every "
            "combination of text representation and classifier: " +
            "; ".join(f"{r.representation} + {r.classifier}: F1 {r.f1_macro_mean:.3f} (std {r.f1_macro_std:.3f}), "
                      f"accuracy {r.accuracy_mean:.3f}, ROC-AUC {r.oof_roc_auc_macro:.3f}"
                      for r in emb_cv.sort_values("f1_macro_mean", ascending=False).itertuples()) + ".")
        pc = emb_per_class.drop(index="macro F1", errors="ignore")
        add("model:embedding_per_class", "model", "Per-class F1: TF-IDF vs MiniLM vs MPNet vs BGE",
            "Out-of-fold F1 per category with Logistic Regression on each representation: " +
            "; ".join(f"{c}: " + ", ".join(f"{col} {pc.loc[c, col]:.2f}" for col in pc.columns) for c in pc.index) +
            ". The two classes TF-IDF missed completely improve the most with BGE: account_access and other.")
        rule = emb_rule.set_index("model")
        add("model:confidence_rule", "model", "Manual review rule: confidence threshold, review share, auto-routing accuracy",
            f"Tickets whose top-class confidence is below {config.CONFIDENCE_THRESHOLD:.0%} go to manual review "
            f"({config.REVIEW_QUEUE}) instead of being routed automatically. " +
            " ".join(f"With {m}: {pct(r['sent to review'])} of tickets are sent to review, auto-routed tickets are "
                     f"{pct(r['auto-routed accuracy'])} correct, reviewed tickets {pct(r['reviewed accuracy'])} correct, "
                     f"overall accuracy {pct(r['overall accuracy'])}." for m, r in rule.iterrows()) +
            " The BGE model cuts the manual review workload by almost two thirds.")
        wrong = emb_changes[~emb_changes["BGE ok"]]
        fixed = emb_changes[~emb_changes["TF-IDF ok"] & emb_changes["BGE ok"]]
        add("model:errors", "model", "Which sentences the models get wrong (error analysis)",
            f"Out of {len(emb_changes)} unique sentences, TF-IDF + LR gets {emb_metrics['messages_right_tfidf']} right "
            f"and BGE + LR gets {emb_metrics['messages_right_bge']} right; BGE fixes {emb_metrics['messages_fixed']} "
            f"sentences and breaks {emb_metrics['messages_broken']}. Fixed by BGE: " +
            "; ".join(f"\"{r.message}\" ({r.true}, TF-IDF said {r._4})" for r in fixed.itertuples()) +
            ". Still wrong with BGE: " +
            "; ".join(f"\"{r.message}\" (true {r.true}, predicted {r._5} with confidence {r._6:.2f})"
                      for r in wrong.itertuples()) +
            ". The TF-IDF model matches words, not meaning: for example the lockout sentence went to security_concern "
            "because 'login' only appeared in a security sentence during training.")

    # ---------------------------------------------------------------- SHAP for the embedding model (03b)
    if shap_checks is not None:
        d = shap_del.set_index("words deleted")
        top_tf = ", ".join(shap_overall[shap_overall["model"] == "TF-IDF + LR"]["term"].head(12))
        top_bge = ", ".join(shap_overall[shap_overall["model"] == "BGE + LR"]["term"].head(12))
        add("model:shap_bge", "model", "How SHAP works for the BGE embedding model (word-level, exact) and is it faithful",
            "The BGE model sees a 768-number vector, so SHAP is computed at the word level: each word is a player, "
            "leaving a word out means deleting it from the sentence and embedding the shorter sentence again, and the "
            "value explained is the model's score (logit) per category. Messages have 8 to 14 words, so every subset of "
            f"words is tried and the Shapley values are exact ({shap_checks['sentences_embedded']} shortened sentences "
            f"embedded for {shap_checks['messages_explained']} messages). Checks: base value plus word values equals the "
            f"model score within {shap_checks['additivity_max_error']:.1e}" +
            (f", and the shap library's Exact explainer agrees within {shap_checks['shap_library_max_difference']:.1e}"
             if shap_checks.get("shap_library_max_difference") is not None else "") + ". " +
            f"Deletion test: removing the top SHAP word lowers the "
            f"predicted-class probability by {d.loc[1, 'top SHAP words']:.3f} on average versus {d.loc[1, 'random words']:.3f} "
            f"for a random word; removing the top 3 lowers it by {d.loc[3, 'top SHAP words']:.3f} versus "
            f"{d.loc[3, 'random words']:.3f} and changes the prediction for {pct(d.loc[3, 'prediction changed (top SHAP)'])} "
            f"of messages. Share of the explanation going to stop words: {pct(shap_checks['stop_word_share_tfidf'])} for "
            f"TF-IDF and {pct(shap_checks['stop_word_share_bge'])} for BGE, while {pct(shap_checks['stop_word_share_of_words'])} "
            f"of words are stop words. Most influential words overall with TF-IDF: {top_tf}. With BGE: {top_bge}.")
        add("model:shap_examples", "model", "Worked examples: TF-IDF vs BGE predictions for new sentences",
            "Predictions for sentences that are not in the dataset: " +
            "; ".join(f"\"{r.message}\": TF-IDF + LR {r._2}, BGE + LR {r._3}" for r in shap_examples.itertuples()) + ".")

    # ---------------------------------------------------------------- clustering with BGE (04b)
    if topics is not None:
        sim = clu_sim
        p8 = clu_scan[clu_scan["k"] == 8].set_index("representation")["purity"]
        add("cluster:bge_vs_lsa", "theme", "Clustering with BGE embeddings vs TF-IDF + LSA: what changed",
            "Notebook 4b repeated the recurring-issue clustering with BGE embeddings. Average cosine similarity of "
            "sentence pairs: " + "; ".join(
                f"{rep}: same problem different product {r.iloc[0]:.2f}, same product different problem {r.iloc[1]:.2f}, "
                f"unrelated {r.iloc[2]:.2f}" for rep, r in sim.iterrows()) +
            ". So BGE also groups sentences about the same product or subject. Clustering all categories into 8 groups "
            "gives a purity of " + ", ".join(f"{k} {v:.2f}" for k, v in p8.items()) +
            ". Within each category, agreement of BGE themes with the notebook 04 themes (adjusted Rand index): " +
            "; ".join(f"{r.category} {r._6:.2f} ({r._3} BGE themes vs {r._4} LSA themes)" for r in clu_cmp.itertuples()) +
            ". BGE merges the login and password failures into one account_access theme, groups vulnerability reports "
            "with suspicious login as security incidents, and moves 'configure data export' into the export theme.")
        add("topics:overview", "theme", "Cross-category topics: subject areas such as login and billing (BGE clustering)",
            f"Clustering all tickets into {len(topics)} groups with BGE embeddings gives subject and intent topics that "
            "cut across categories: " + "; ".join(
                f"topic {r.topic} ({r._2}): {r.tickets} tickets, categories {r.categories}, e.g. \"{r.example}\""
                for r in topics.itertuples()) +
            ". The login topic collects all account_access tickets plus login auth bugs, slowness, set-up and "
            "vulnerability tickets; the billing topic collects billing_problem plus billing bugs, set-up and vulnerability "
            "tickets. Reporting by category alone hides these subject areas.")

    # ---------------------------------------------------------------- recurring themes
    for r in themes.itertuples():
        add(f"theme:{r.cluster_id}", "theme", f"Recurring issue in {r.category}: {r.theme}",
            f"Recurring issue theme in {r.category} (notebook 04) with top words {r.theme}. It covers {r.tickets} tickets "
            f"({pct(r.share_of_category)} of the category) from {r.unique_sentences} different descriptions; cohesion "
            f"{r.cohesion:.2f}. Example description: \"{r.example}\".")

    # ---------------------------------------------------------------- anomalies
    flags = volume[volume["is_anomaly"] == True]
    flag_txt = "; ".join(f"{r.category} in period starting {str(r.period)[:10]}: {int(r['count'])} tickets vs "
                         f"{r.expected:.0f} expected (z = {r.z:.1f})" for _, r in flags.iterrows()) or "none"
    add("anomaly:volume", "anomaly", "Volume anomalies (spikes and drops)",
        f"Ticket volume was checked weekly overall and monthly per category with a robust z-score (threshold 3). "
        f"Flagged periods: {flag_txt}. Detector check with planted spikes: " +
        "; ".join(f"{r.check}: {r.result}" for r in validation.itertuples()) +
        ". The number of real flags is in line with chance, so there are no strong demand spikes.")
    sig = hot[hot["significant"] == True].sort_values("z", ascending=False)
    add("anomaly:hotspots", "anomaly", "Hotspots: groups needing operational attention",
        "Segments whose rate is significantly above the overall rate (Bonferroni corrected): " +
        "; ".join(f"{r.metric.replace('_', ' ')} for {r.group} = {r.value}: {pct(r.rate)} vs {pct(r.overall)} overall"
                  for r in sig.itertuples()) +
        ". No channel, region, product, platform, customer segment or SLA plan stands out.")
    add("anomaly:tickets", "anomaly", "Unusual individual tickets (Isolation Forest)",
        f"Isolation Forest flagged {len(odd)} resolved tickets as unusual. The most common reasons: " +
        "; ".join(f"{k} ({v} tickets)" for k, v in odd["why"].value_counts().head(4).items()) +
        f". Flagged tickets were resolved in {odd['resolution_time_hours'].mean():.1f} hours on average, faster than "
        f"usual, which suggests some tickets are closed too quickly and then reopened.")

    add("data:quality", "data", "Data quality, cleaning and assumptions",
        "Data quality checks found no duplicate tickets, no empty messages and no impossible values. Resolution fields "
        "are empty only for unfinished tickets and are left empty. About 20% of region values were missing at random and "
        "were set to Unknown. A CSAT score of 0 means not rated and is treated as missing. The data is synthetic: "
        "timestamps are generated, the routing table from category to support queue is an assumption because the data "
        "has no assignment group, and resolution times never exceed a fixed limit per priority.")

    docs += narrative_docs()
    return pd.DataFrame(docs)


# ------------------------------------------------------------------ narrative documents
def _sections(markdown, min_chars=200):
    """Split markdown into (heading, text) sections at '## ' headings."""
    out, heading, buf = [], "Introduction", []
    for line in markdown.split("\n"):
        if line.startswith("## "):
            out.append((heading, "\n".join(buf)))
            heading, buf = line[3:].strip(), []
        elif not line.startswith("# "):
            buf.append(line)
    out.append((heading, "\n".join(buf)))
    return [(h, re.sub(r"\n{3,}", "\n\n", b).strip()) for h, b in out if len(b.strip()) >= min_chars]


def narrative_docs(root=config.ROOT):
    """The written interpretations: every notebook's markdown, the evaluation report and the README, one doc per section."""
    docs = []
    for nb_path in sorted((root / "notebooks").glob("*.ipynb")):
        nb = json.load(open(nb_path))
        md = "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
        first = next((l for l in md.split("\n") if l.startswith("# ")), f"# {nb_path.stem}")[2:].strip()
        for i, (heading, text) in enumerate(_sections(md)):
            if "save" in heading.lower():
                continue
            docs.append({"doc_id": f"notebook:{nb_path.stem.split('_')[0]}#{i}", "source": "notebook",
                         "title": f"{first}: {heading}", "text": mask(text)})
    for path, prefix, label, skip in [(root / "reports" / "evaluation_report.md", "report", "Evaluation report", ()),
                                      (root / "README.md", "readme", "Project README",
                                       ("folder structure", "brief deliverables", "reproducibility"))]:
        if path.exists():
            for i, (heading, text) in enumerate(_sections(path.read_text())):
                if heading.lower().startswith(skip):
                    continue
                docs.append({"doc_id": f"{prefix}#{i}", "source": prefix, "title": f"{label}: {heading}",
                             "text": mask(text)})
    return docs


# ------------------------------------------------------------------ retrieval
# question words that say what to do, not what to look for
QUESTION_WORDS = {"explain", "tell", "describe", "show", "give", "summarise", "summarize", "please", "what", "which",
                  "why", "how", "does", "did", "category", "categories"}
STOP_WORDS = list(ENGLISH_STOP_WORDS | QUESTION_WORDS)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "     # BGE's instruction for queries


def normalise(text):
    return str(text).lower().replace("_", " ")          # "billing_problem" should match "billing problem"


def load_encoder():
    """BGE sentence encoder, or None if sentence-transformers isn't installed."""
    try:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(config.EMBEDDING_MODEL)
    except Exception:
        return None


class Retriever:
    """Hybrid search: TF-IDF keyword similarity, plus BGE semantic similarity when an encoder is available.
    Each score is scaled by its best value for the question, then combined (semantic weight `alpha`)."""

    def __init__(self, kb, encoder="auto", alpha=0.8):
        self.kb = kb.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(stop_words=STOP_WORDS, ngram_range=(1, 2), sublinear_tf=True,
                                          preprocessor=normalise)
        self.matrix = self.vectorizer.fit_transform(self.kb["title"] + ". " + self.kb["title"] + ". " + self.kb["text"])
        self.encoder = load_encoder() if encoder == "auto" else encoder
        self.alpha = alpha if self.encoder is not None else 0.0
        self.dense = None
        if self.encoder is not None:
            self.dense = self.encoder.encode((self.kb["title"] + ". " + self.kb["text"].str.slice(0, 1500)).tolist(),
                                             batch_size=32, normalize_embeddings=True, show_progress_bar=False)

    @property
    def mode(self):
        return "hybrid (keywords + BGE semantic search)" if self.alpha > 0 else "keywords (TF-IDF)"

    def scores(self, question):
        kw = cosine_similarity(self.vectorizer.transform([question]), self.matrix).ravel()
        if self.alpha == 0:
            return kw
        sem = self.dense @ self.encoder.encode([QUERY_PREFIX + question], normalize_embeddings=True,
                                               show_progress_bar=False)[0]
        scale = lambda s: s / s.max() if s.max() > 0 else s
        return (1 - self.alpha) * scale(kw) + self.alpha * scale(np.clip(sem, 0, None))

    def search(self, question, k=5):
        scores = self.scores(question)
        top = np.argsort(-scores)[:k]
        return self.kb.iloc[top].assign(score=scores[top])


# ------------------------------------------------------------------ generation
def llm_provider():
    if os.getenv("MISTRAL_API_KEY"):
        return "mistral"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "offline"


def _post(url, headers, payload, retries=2):
    """POST with a short back-off on rate limits (HTTP 429)."""
    for attempt in range(retries + 1):
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code != 429 or attempt == retries:
            return r
        time.sleep(1.5 * (attempt + 1))


def call_llm(messages, provider, max_tokens=600):
    """messages: list of {"role": "user"|"assistant", "content": ...}. Returns (text, model name)."""
    if provider == "anthropic":
        r = _post("https://api.anthropic.com/v1/messages",
                  {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
                   "content-type": "application/json"},
                  {"model": config.ANTHROPIC_MODEL, "max_tokens": max_tokens, "temperature": 0,
                   "system": SYSTEM_PROMPT, "messages": messages})
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"]).strip(), config.ANTHROPIC_MODEL

    if provider == "mistral":
        base, key, models = config.MISTRAL_BASE_URL, os.environ["MISTRAL_API_KEY"], config.MISTRAL_MODELS
    else:
        base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        key, models = os.environ["OPENAI_API_KEY"], [os.getenv("OPENAI_MODEL", config.OPENAI_MODEL)]
    last = None
    for model in models:                                   # fall back to the next model if one is unavailable
        r = _post(f"{base}/chat/completions", {"Authorization": f"Bearer {key}"},
                  {"model": model, "temperature": 0, "max_tokens": max_tokens,
                   "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages})
        if r.ok:
            return r.json()["choices"][0]["message"]["content"].strip(), model
        last = r
        if r.status_code not in (429, 403, 404):
            break
    last.raise_for_status()


def offline_answer(question, docs):
    """No LLM: answer with the most relevant document, plus the best sentences of the runner-up if it is nearly as relevant."""
    if docs.empty or docs["score"].iloc[0] == 0:
        return "I couldn't find this in the project results."
    top = docs.iloc[0]
    parts = [f"{top.text} [{top.doc_id}]"]
    if len(docs) > 1 and docs["score"].iloc[1] >= 0.6 * top.score:
        second = docs.iloc[1]
        sentences = [s for s in re.split(r"(?<=\.)\s+(?=[A-Z])", second.text) if s.strip()]
        vec = TfidfVectorizer(stop_words="english", preprocessor=normalise).fit(sentences + [question])
        sims = cosine_similarity(vec.transform([question]), vec.transform(sentences)).ravel()
        best = [sentences[i] for i in sorted(np.argsort(-sims)[:2]) if sims[i] > 0]
        if best:
            parts.append(" ".join(best) + f" [{second.doc_id}]")
    return "\n\n".join(parts)


def retrieve(question, retriever, k=5, history=None):
    """Search with the question; for follow-ups also with the previous question, and merge the results."""
    docs = retriever.search(question, k)
    prev = [m["content"] for m in (history or []) if m["role"] == "user"]
    if prev:
        extra = retriever.search(prev[-1] + " " + question, k)
        docs = (pd.concat([docs.head(k - 2), extra]).drop_duplicates("doc_id").head(k))
    return docs


def answer(question, retriever, k=5, history=None):
    """Answer one question. history: earlier chat turns as [{"role", "content"}], used for follow-up questions."""
    docs = retrieve(question, retriever, k, history)
    context = "\n\n".join(f"[{d.doc_id}] {d.title}\n{d.text}" for d in docs.itertuples())
    provider = llm_provider()
    if provider != "offline":
        turns = [{"role": m["role"], "content": mask(m["content"])[:1500]} for m in (history or [])[-6:]]
        prompt = f"Context documents:\n{context}\n\nQuestion: {mask(question)}"
        try:
            text, model = call_llm(turns + [{"role": "user", "content": prompt}], provider)
            mode = f"LLM ({provider}: {model})"
        except Exception as e:
            text, mode = offline_answer(question, docs), f"offline (LLM error: {type(e).__name__})"
    else:
        text, mode = offline_answer(question, docs), "offline (no API key set)"
    return {"question": question, "answer": text, "mode": mode,
            "sources": list(docs["doc_id"]), "titles": list(docs["title"]), "scores": list(docs["score"].round(3)),
            "numbers_grounded": numbers_grounded(text, context)}


def explain_prediction(text, label, confidence, shap_terms, retriever, model_name="the classifier"):
    """Plain-English explanation of one prediction, combining SHAP words with retrieved category facts."""
    words_for = [t for t, v in shap_terms if v > 0][:4]
    words_against = [t for t, v in shap_terms if v < 0][:2]
    facts = retriever.search(f"Category {label}", k=1).iloc[0]
    if llm_provider() != "offline":
        prompt = (f"Context documents:\n[{facts.doc_id}] {facts.text}\n\nA ticket said: \"{mask(text)}\". "
                  f"{model_name} predicted {label} with {confidence:.0%} confidence (manual review below "
                  f"{config.CONFIDENCE_THRESHOLD:.0%}). SHAP: words pushing towards {label}: {words_for}; "
                  f"words pushing against: {words_against}. Explain the prediction to a support agent in 3 sentences.")
        try:
            return call_llm([{"role": "user", "content": prompt}], llm_provider(), max_tokens=300)[0], "LLM"
        except Exception:
            pass
    msg = (f"The model predicted {label} with {confidence:.0%} confidence. The words that pushed it there most were "
           f"{', '.join(repr(w) for w in words_for) or 'none'}")
    msg += f", while {', '.join(repr(w) for w in words_against)} pulled the other way. " if words_against else ". "
    msg += ("Confidence is below the review threshold, so a person should confirm the category. "
            if confidence < config.CONFIDENCE_THRESHOLD else "Confidence is high enough to route it automatically "
            f"to {config.ROUTING.get(label, 'the right team')}. ")
    return msg, "offline"
