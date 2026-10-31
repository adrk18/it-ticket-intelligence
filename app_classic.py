"""IT Support Ticket Analytics Dashboard

Run the six notebooks first (they create everything in data/processed and models), then:
    streamlit run app.py
"""
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from src import config, rag

st.set_page_config(page_title="IT Ticket Analytics", page_icon="🎫", layout="wide")

# Streamlit >= 1.48 uses width="stretch"; older versions use use_container_width=True
_version = tuple(int(x) for x in st.__version__.split(".")[:2])
W = {"width": "stretch"} if _version >= (1, 48) else {"use_container_width": True}

P = config.PROCESSED
CLASSES = sorted(config.ROUTING)
COLORS = dict(zip(CLASSES, px.colors.qualitative.Set2))


# ---------------------------------------------------------------- data loading
@st.cache_data
def load_data():
    tickets = pd.read_csv(P / "tickets_clean.csv", parse_dates=["created_at"])
    preds = pd.read_csv(P / "predictions.csv").drop(columns=["true_category"])
    clusters = pd.read_csv(P / "clusters.csv")
    view = (tickets.merge(preds, on="ticket_id", how="left", validate="one_to_one")
                   .merge(clusters[["ticket_id", "cluster_id", "theme"]], on="ticket_id", how="left",
                          validate="one_to_one"))
    files = {name: pd.read_csv(P / f"{name}.csv") for name in
             ["model_comparison", "cv_scores", "classification_report", "shap_global", "shap_local",
              "cluster_summary", "volume_anomalies", "hotspots", "ticket_anomalies", "anomaly_validation",
              "knowledge_base", "rag_retrieval_eval"]}
    files["metrics"] = json.load(open(P / "metrics.json"))
    return view, files


@st.cache_resource
def load_model():
    model = joblib.load(config.MODELS / "tfidf_logreg.joblib")
    background = np.load(config.MODELS / "shap_background.npy")
    return model, background


@st.cache_resource
def load_retriever(kb):
    return rag.Retriever(kb)


try:
    data, F = load_data()
    model, background = load_model()
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}. Run notebooks 01 to 06 first; they create data/processed and models.")
    st.stop()
retriever = load_retriever(F["knowledge_base"])


def style(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=45, b=10), template="plotly_white",
                      legend=dict(orientation="h", y=-0.2))
    return fig


# ---------------------------------------------------------------- sidebar filters
st.sidebar.title("🎫 Filters")
dmin, dmax = data["created_at"].min().date(), data["created_at"].max().date()
dates = st.sidebar.date_input("Created between", (dmin, dmax), min_value=dmin, max_value=dmax)
if not (isinstance(dates, (list, tuple)) and len(dates) == 2):
    dates = (dmin, dmax)
f_cat = st.sidebar.multiselect("Category", CLASSES)
f_pri = st.sidebar.multiselect("Priority", ["urgent", "high", "medium", "low"])
f_chan = st.sidebar.multiselect("Channel", sorted(data["channel"].unique()))

mask = data["created_at"].dt.date.between(dates[0], dates[1])
if f_cat:
    mask &= data[config.TARGET].isin(f_cat)
if f_pri:
    mask &= data["priority"].isin(f_pri)
if f_chan:
    mask &= data["channel"].isin(f_chan)
v = data[mask]

st.sidebar.metric("Tickets in view", f"{len(v):,}")
ok = v["ticket_id"].is_unique and v[config.TARGET].value_counts().sum() == len(v)
st.sidebar.caption(("✅" if ok else "⚠️") + " counts check: one row per ticket, category counts add up to the total")
st.sidebar.caption(f"Data: stratified sample of {len(data):,} synthetic tickets. Routing table is assumed.")

if v.empty:
    st.warning("No tickets match these filters.")
    st.stop()

st.title("IT Support Ticket Analytics")
tabs = st.tabs(["📊 Overview", "🔮 Classify a ticket", "📈 Model performance", "🧠 Explainability",
                "🔁 Recurring issues", "🚨 Anomalies", "🤖 GenAI assistant", "🗂️ Data & assumptions"])

# ================================================================ Overview
with tabs[0]:
    c = st.columns(6)
    c[0].metric("Tickets", f"{len(v):,}")
    c[1].metric("Resolved", f"{v['status'].eq('resolved').mean():.0%}")
    c[2].metric("Open backlog", f"{v['status'].isin(['open', 'in_progress', 'on_hold']).sum():,}")
    c[3].metric("Negative customers", f"{v['customer_sentiment'].isin(['negative', 'very_negative']).mean():.0%}")
    c[4].metric("Median resolution", f"{v.loc[v['status'] == 'resolved', 'resolution_time_hours'].median():.1f} h")
    c[5].metric("Sent to manual review", f"{v['needs_review'].mean():.0%}")

    left, right = st.columns(2)
    counts = v[config.TARGET].value_counts().rename_axis("category").reset_index(name="tickets")
    fig = px.bar(counts, x="tickets", y="category", orientation="h", color="category", color_discrete_map=COLORS,
                 text="tickets", title="Tickets per category")
    left.plotly_chart(style(fig).update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"}), **W)

    queues = v["queue"].value_counts().rename_axis("queue").reset_index(name="tickets")
    fig = px.bar(queues, x="tickets", y="queue", orientation="h", text="tickets",
                 title="Tickets per support queue (predicted)")
    right.plotly_chart(style(fig).update_layout(yaxis={"categoryorder": "total ascending"}), **W)
    right.caption("Low-confidence predictions go to Manual Triage. The category → queue table is an assumption.")

    monthly = v.groupby([pd.Grouper(key="created_at", freq="MS"), config.TARGET]).size().rename("tickets").reset_index()
    fig = px.line(monthly, x="created_at", y="tickets", color=config.TARGET, color_discrete_map=COLORS,
                  title="Tickets per month by category")
    st.plotly_chart(style(fig, 400).update_yaxes(rangemode="tozero"), **W)

    left, right = st.columns(2)
    res = v[v["status"] == "resolved"]
    trend = res.groupby([pd.Grouper(key="created_at", freq="QS"), "priority"])["resolution_time_hours"] \
               .median().rename("median_hours").reset_index()
    fig = px.line(trend, x="created_at", y="median_hours", color="priority", markers=True,
                  category_orders={"priority": ["urgent", "high", "medium", "low"]},
                  title="Resolution trend: median hours per quarter, by priority")
    left.plotly_chart(style(fig).update_yaxes(rangemode="tozero"), **W)

    sat = v.groupby(config.TARGET).agg(negative=("customer_sentiment",
                                                 lambda s: s.isin(["negative", "very_negative"]).mean()),
                                       csat=("csat_score", "mean")).reset_index().sort_values("negative")
    fig = px.bar(sat, x="negative", y=config.TARGET, orientation="h", color=config.TARGET, color_discrete_map=COLORS,
                 hover_data={"csat": ":.2f"}, title="Share of negative customers by category")
    right.plotly_chart(style(fig).update_layout(showlegend=False, xaxis_tickformat=".0%"), **W)

# ================================================================ Classify
with tabs[1]:
    st.subheader("Predict the issue type of a new ticket")
    st.caption("TF-IDF + Logistic Regression, trained on all sample tickets. Below 60% confidence → manual review.")
    examples = {"Common": "I was charged twice for my subscription this month.",
                "New wording": "Locked out of my account after entering the wrong password too many times",
                "Ambiguous": "How do I reset my password?",
                "Out of scope": "The coffee machine on the third floor is broken."}
    cols = st.columns(len(examples))
    for col, (name, text) in zip(cols, examples.items()):
        if col.button(name, **W):
            st.session_state["ticket"] = text
    text = st.text_area("Ticket description", key="ticket", height=90, placeholder="Describe the problem...")

    if text.strip():
        proba = model.predict_proba([text])[0]
        order = np.argsort(-proba)
        label, conf = model.classes_[order[0]], proba[order[0]]
        review = conf < config.CONFIDENCE_THRESHOLD

        c = st.columns(3)
        c[0].metric("Predicted category", label)
        c[1].metric("Confidence", f"{conf:.0%}")
        c[2].metric("Route to", config.REVIEW_QUEUE if review else config.ROUTING[label])
        if review:
            st.warning("Low confidence: this ticket goes to manual review instead of being routed automatically.")
        else:
            st.success("Confident prediction: routed automatically.")

        tfidf = model.named_steps["tfidfvectorizer"]
        clf = model.named_steps["logisticregression"]
        x = tfidf.transform([text]).toarray()[0]
        contrib = (x - background) * clf.coef_[order[0]]
        present = np.where(x > 0)[0]
        terms = [(tfidf.get_feature_names_out()[j], float(contrib[j]))
                 for j in present[np.argsort(-np.abs(contrib[present]))][:8]]

        left, right = st.columns(2)
        top = pd.DataFrame({"category": model.classes_[order[:5]], "probability": proba[order[:5]]})
        fig = px.bar(top, x="probability", y="category", orientation="h", color="category",
                     color_discrete_map=COLORS, title="Top 5 categories")
        left.plotly_chart(style(fig, 320).update_layout(showlegend=False, xaxis_range=[0, 1], xaxis_tickformat=".0%",
                                                        yaxis={"autorange": "reversed"}), **W)
        if terms:
            sh = pd.DataFrame(terms, columns=["word", "shap"])
            sh["direction"] = np.where(sh["shap"] > 0, f"towards {label}", f"against {label}")
            fig = px.bar(sh, x="shap", y="word", orientation="h", color="direction",
                         title="Why? Word contributions (SHAP)")
            right.plotly_chart(style(fig, 320).update_layout(yaxis={"autorange": "reversed"}), **W)
        else:
            right.info("None of these words were seen in training, so there's nothing to explain. "
                       "The ticket is likely out of scope.")
        if st.button("✨ Explain with GenAI"):
            explanation, mode = rag.explain_prediction(text, label, conf, terms, retriever)
            st.info(f"**{'AI-generated' if mode == 'LLM' else 'Auto-generated (offline)'}:** {explanation}")

# ================================================================ Model performance
with tabs[2]:
    m = F["metrics"]
    st.subheader("How good is the classifier?")
    st.caption(f"{m['model']}. Evaluated with {m['evaluation']}: every ticket is predicted by a model that never saw its "
               "message, so these numbers reflect performance on new wording.")
    c = st.columns(6)
    c[0].metric("Macro F1", f"{m['macro_f1']:.3f}")
    c[1].metric("Weighted F1", f"{m['weighted_f1']:.3f}")
    c[2].metric("Macro precision", f"{m['macro_precision']:.3f}")
    c[3].metric("Macro recall", f"{m['macro_recall']:.3f}")
    c[4].metric("ROC-AUC (macro)", f"{m['roc_auc_macro']:.3f}")
    c[5].metric("Accuracy", f"{m['accuracy']:.3f}")

    left, right = st.columns(2)
    left.markdown("**Model comparison** (test set of unseen messages)")
    left.dataframe(F["model_comparison"].round(3), hide_index=True, **W)
    right.markdown("**Grouped 5-fold cross-validation**")
    right.dataframe(F["cv_scores"].round(3), hide_index=True, **W)

    # metrics recomputed live for the filtered tickets
    e = v.dropna(subset=["pred_category"])
    p, r, f1, s = precision_recall_fscore_support(e[config.TARGET], e["pred_category"], labels=CLASSES, zero_division=0)
    per = pd.DataFrame({"category": CLASSES, "precision": p, "recall": r, "f1": f1, "tickets": s})
    per["roc_auc"] = [roc_auc_score((e[config.TARGET] == c_).astype(int), e[f"proba_{c_}"])
                      if 0 < (e[config.TARGET] == c_).sum() < len(e) else np.nan for c_ in CLASSES]
    left, right = st.columns([1, 1])
    left.markdown(f"**Per-class scores** ({len(e):,} tickets in the current filter)")
    left.dataframe(per.round(3), hide_index=True, **W)

    cm = pd.crosstab(e[config.TARGET], e["pred_category"]).reindex(index=CLASSES, columns=CLASSES, fill_value=0)
    normalise = right.toggle("Show as % of each actual category", value=True)
    shown = cm.div(cm.sum(axis=1).replace(0, 1), axis=0) if normalise else cm
    fig = px.imshow(shown, text_auto=".0%" if normalise else "d", color_continuous_scale="Blues", aspect="auto",
                    labels=dict(x="Predicted", y="Actual", color=""), title="Confusion matrix")
    right.plotly_chart(style(fig, 450), **W)

    left, right = st.columns(2)
    e = e.assign(result=np.where(e["correct"], "correct", "wrong"))
    fig = px.histogram(e, x="confidence", color="result", nbins=20, barmode="stack",
                       color_discrete_map={"correct": "#4c78a8", "wrong": "#e45756"},
                       title="Confidence of correct and wrong predictions")
    fig.add_vline(x=config.CONFIDENCE_THRESHOLD, line_dash="dash")
    left.plotly_chart(style(fig), **W)

    threshold = right.slider("Try a different review threshold", 0.3, 0.95, config.CONFIDENCE_THRESHOLD, 0.05)
    auto, review = e[e["confidence"] >= threshold], e[e["confidence"] < threshold]
    c = right.columns(2)
    c[0].metric("Routed automatically", f"{len(auto) / len(e):.0%}",
                f"accuracy {auto['correct'].mean():.0%}" if len(auto) else None, delta_color="off")
    c[1].metric("Sent to review", f"{len(review) / len(e):.0%}",
                f"accuracy {review['correct'].mean():.0%}" if len(review) else None, delta_color="off")
    right.caption("A useful threshold sends the uncertain (often wrong) tickets to review while keeping "
                  "auto-routed accuracy high.")

# ================================================================ Explainability
with tabs[3]:
    st.subheader("What drives each prediction? (SHAP)")
    sg = F["shap_global"]
    cat = st.selectbox("Category", CLASSES, key="shap_cat")
    d = sg[sg["category"] == cat].sort_values("mean_shap")
    fig = px.bar(d, x="mean_shap", y="term", orientation="h", color_discrete_sequence=[COLORS[cat]],
                 title=f"Words that push tickets into '{cat}' (mean SHAP value)")
    st.plotly_chart(style(fig, 420), **W)
    st.markdown("**Explain one description**")
    sl = F["shap_local"]
    msg = st.selectbox("Description", sorted(sl[config.TEXT].unique()), key="shap_msg")
    d = sl[sl[config.TEXT] == msg]
    st.write(f"Predicted **{d['predicted'].iloc[0]}** with {d['confidence'].iloc[0]:.0%} confidence")
    fig = px.bar(d.sort_values("shap"), x="shap", y="term", orientation="h",
                 color=np.where(d.sort_values("shap")["shap"] > 0, "towards", "against"),
                 title="Word contributions")
    st.plotly_chart(style(fig, 340), **W)

# ================================================================ Recurring issues
with tabs[4]:
    st.subheader("Recurring issue themes (text embeddings + K-Means)")
    summary = F["cluster_summary"]
    counts = v["cluster_id"].value_counts().rename("tickets_in_view")
    themes = summary.merge(counts, left_on="cluster_id", right_index=True, how="left").fillna({"tickets_in_view": 0})
    themes["label"] = themes["category"] + ": " + themes["theme"].str.split(", ").str[:2].str.join(", ")
    fig = px.bar(themes.sort_values("tickets_in_view").tail(15), x="tickets_in_view", y="label", orientation="h",
                 color="category", color_discrete_map=COLORS, title="Largest themes in the current filter")
    st.plotly_chart(style(fig, 480).update_layout(showlegend=False), **W)

    pick = st.selectbox("Look at a theme", themes.sort_values("tickets_in_view", ascending=False)["label"])
    row = themes[themes["label"] == pick].iloc[0]
    c = st.columns(3)
    c[0].metric("Tickets", f"{int(row['tickets_in_view']):,}")
    c[1].metric("Different descriptions", int(row["unique_sentences"]))
    c[2].metric("Cohesion", f"{row['cohesion']:.2f}")
    st.markdown("**Representative descriptions**")
    for text_ in v.loc[v["cluster_id"] == row["cluster_id"], config.TEXT].value_counts().head(5).index:
        st.write("•", text_)
    tr = v[v["cluster_id"] == row["cluster_id"]].groupby(pd.Grouper(key="created_at", freq="QS")).size() \
         .rename("tickets").reset_index()
    fig = px.line(tr, x="created_at", y="tickets", markers=True, title="Theme volume per quarter")
    st.plotly_chart(style(fig, 300).update_yaxes(rangemode="tozero"), **W)

# ================================================================ Anomalies
with tabs[5]:
    st.subheader("Anomaly detection")
    va = F["volume_anomalies"].copy()
    va["period"] = pd.to_datetime(va["period"])
    series = st.selectbox("Series", ["ALL (weekly)"] + [f"{c_} (monthly)" for c_ in CLASSES])
    key = series.split(" ")[0]
    s = va[va["category"] == key]
    fig = go.Figure([
        go.Scatter(x=s["period"], y=s["expected"], name="expected", line=dict(dash="dash", color="grey")),
        go.Scatter(x=s["period"], y=s["count"], name="actual", line=dict(color="#4c78a8")),
        go.Scatter(x=s.loc[s["is_anomaly"], "period"], y=s.loc[s["is_anomaly"], "count"], mode="markers",
                   name="anomaly", marker=dict(color="red", size=12)),
    ])
    st.plotly_chart(style(fig, 360).update_layout(title=f"Ticket volume: {series}").update_yaxes(rangemode="tozero"), **W)
    left, right = st.columns(2)
    left.markdown("**Flagged periods**")
    left.dataframe(va[va["is_anomaly"]][["period", "category", "count", "expected", "z"]].round(1), hide_index=True, **W)
    right.markdown("**Detector check** (planted spikes in a copy of the data)")
    right.dataframe(F["anomaly_validation"], hide_index=True, **W)

    st.markdown("**Hotspots: groups significantly worse than average** (Bonferroni-corrected)")
    hs = F["hotspots"]
    st.dataframe(hs[hs["significant"]].sort_values("z", ascending=False)
                 .style.format({"rate": "{:.1%}", "overall": "{:.1%}", "z": "{:.1f}"}), hide_index=True, **W)

    st.markdown("**Unusual individual tickets** (Isolation Forest on resolved tickets)")
    ta = F["ticket_anomalies"]
    ta = ta[ta["ticket_id"].isin(v["ticket_id"])]
    left, right = st.columns([1, 2])
    pat = ta["why"].value_counts().head(6).rename_axis("pattern").reset_index(name="tickets")
    fig = px.bar(pat, x="tickets", y="pattern", orientation="h", title="Most common patterns")
    left.plotly_chart(style(fig, 330).update_layout(yaxis={"categoryorder": "total ascending"}), **W)
    right.dataframe(ta.sort_values("anomaly_score", ascending=False).round(2), hide_index=True, **W)

# ================================================================ GenAI assistant
with tabs[6]:
    st.subheader("Ask the data (RAG assistant)")
    provider = rag.llm_provider()
    st.caption(f"Answers are built only from this project's results. Mode: "
               f"{'LLM via ' + provider if provider != 'offline' else 'offline (set ANTHROPIC_API_KEY or OPENAI_API_KEY for LLM answers)'}.")
    suggestions = ["Which categories have the most unhappy customers?", "How good is the model?",
                   "What are the recurring bug issues?", "Were there any spikes in ticket volume?",
                   "Which groups need operational attention?", "Is there data leakage?"]
    cols = st.columns(3)
    for i, q in enumerate(suggestions):
        if cols[i % 3].button(q, key=f"sugg{i}", **W):
            st.session_state["question"] = q
    question = st.text_input("Your question", key="question")
    if question.strip():
        result = rag.answer(question, retriever)
        st.markdown(result["answer"])
        if not result["numbers_grounded"]:
            st.warning("Some numbers in this answer were not found in the source documents. Please double-check.")
        st.caption(f"Mode: {result['mode']}")
        kb = F["knowledge_base"].set_index("doc_id")
        for doc_id, score in zip(result["sources"], result["scores"]):
            with st.expander(f"Source: {kb.loc[doc_id, 'title']} (relevance {score:.2f})"):
                st.write(kb.loc[doc_id, "text"])

# ================================================================ Data & assumptions
with tabs[7]:
    st.subheader("Data and assumptions")
    st.markdown("""
- **Data**: a stratified sample of 20,000 tickets from a synthetic Kaggle dataset of 100,000 IT support tickets (seed 42).
- **Target**: `issue_type`, 8 balanced categories. **Model input**: the ticket message only.
- **Leakage handled**: `agent_first_reply` and `resolution_summary` reveal the label and are never used; train/test splits are made by message.
- **Cleaning**: `csat_score = 0` means not rated and is treated as missing; missing region is "Unknown"; resolution fields are empty for unfinished tickets.
- **Assumed**: timestamps are synthetic; the category → support queue table is our own assumption (no assignment-group data).
- **Out of scope**: the system suggests categories and routing; it never closes or changes tickets.
""")
    st.markdown("**Assistant retrieval test**")
    st.dataframe(F["rag_retrieval_eval"], hide_index=True, **W)
    st.markdown("**Counts check for the current filter**")
    st.dataframe(pd.DataFrame({
        "check": ["one row per ticket", "category counts add up to total", "queue counts add up to total"],
        "passed": [v["ticket_id"].is_unique, v[config.TARGET].value_counts().sum() == len(v),
                   v["queue"].value_counts().sum() == len(v)],
    }), hide_index=True, **W)
