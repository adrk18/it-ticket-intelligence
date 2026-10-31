"""Ticket Intelligence: IT support analytics, live ticket classification and an AI assistant.

Run the notebooks first (they create data/processed and models), then:
    .venv/bin/streamlit run app.py
The Mistral key is read from .env (see src/rag.py). The previous dashboard is kept in app_classic.py.
"""
import json
import os

os.environ.update(TQDM_DISABLE="1", TRANSFORMERS_VERBOSITY="error", HF_HUB_DISABLE_PROGRESS_BARS="1",
                  TOKENIZERS_PARALLELISM="false")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from sklearn.metrics import roc_auc_score

from src import config, rag
from src.inference import TicketClassifier, highlight_html, load_encoder

st.set_page_config(page_title="Ticket Intelligence", page_icon="🎫", layout="wide", initial_sidebar_state="expanded")

W = {"width": "stretch"}
P = config.PROCESSED
T = config.TEXT
Y = config.TARGET
CLASSES = sorted(config.ROUTING)
COLORS = {"account_access": "#6366F1", "billing_problem": "#F59E0B", "bug": "#EF4444", "feature_request": "#10B981",
          "how_to": "#06B6D4", "other": "#94A3B8", "performance": "#8B5CF6", "security_concern": "#EC4899"}
LABEL = {c: c.replace("_", " ").capitalize() for c in CLASSES}
MODEL_COLORS = {"BGE + LR": "#4F46E5", "TF-IDF + LR": "#A1A1AA"}
PRIORITIES = ["urgent", "high", "medium", "low"]

# ------------------------------------------------------------------ look and feel
st.markdown("""
<style>
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1450px; }
h1, h2, h3 { letter-spacing: -0.02em; }
.hero { background: linear-gradient(120deg, #4F46E5 0%, #7C3AED 55%, #06B6D4 100%); border-radius: 22px;
        padding: 26px 32px; color: #fff; box-shadow: 0 12px 32px rgba(79,70,229,.22); margin-bottom: 1.2rem; }
.hero h1 { color: #fff; font-size: 1.85rem; font-weight: 800; margin: 0; padding: 0; }
.hero p { color: rgba(255,255,255,.92); margin: .4rem 0 0 0; font-size: 1rem; }
.hero .tag { display: inline-block; background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.3);
             padding: 3px 12px; border-radius: 999px; font-size: .78rem; font-weight: 600; margin-right: 6px; margin-top: 10px; }
.kpi { background: #fff; border: 1px solid #ECEEF5; border-radius: 16px; padding: 14px 18px 16px 18px;
       box-shadow: 0 2px 12px rgba(17,24,39,.04); height: 100%; min-height: 142px; }
.kpi .bar { height: 4px; width: 42px; border-radius: 4px; margin-bottom: 10px; }
.kpi .label { color: #6B7280; font-size: .74rem; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; }
.kpi .value { font-size: 1.65rem; font-weight: 800; color: #111827; margin-top: 2px; line-height: 1.2; }
.kpi .sub { font-size: .8rem; color: #6B7280; margin-top: 4px; }
.kpi .up { color: #059669; font-weight: 700; } .kpi .down { color: #DC2626; font-weight: 700; }
.pred { border-radius: 20px; padding: 22px 24px; color: #fff; box-shadow: 0 12px 28px rgba(17,24,39,.15); }
.pred .small { font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; opacity: .85; font-weight: 700; }
.pred .big { font-size: 2rem; font-weight: 800; margin: 2px 0 6px 0; }
.pill { display: inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 700; font-size: .8rem;
        background: rgba(255,255,255,.2); border: 1px solid rgba(255,255,255,.35); margin: 4px 6px 0 0; }
.chip { display: inline-block; padding: 3px 10px; border-radius: 999px; font-weight: 600; font-size: .78rem; color: #fff; }
.insight { background: #fff; border: 1px solid #ECEEF5; border-left: 5px solid #4F46E5; border-radius: 12px;
           padding: 12px 16px; margin-bottom: 10px; font-size: .92rem; color: #1F2937; }
.textbox { background: #fff; border: 1px solid #ECEEF5; border-radius: 14px; padding: 14px 16px; line-height: 2.1; font-size: 1.02rem; }
.section { font-size: 1.15rem; font-weight: 800; color: #111827; margin: 1.1rem 0 .4rem 0; }
.muted { color: #6B7280; font-size: .85rem; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #F5F3FF 0%, #FFFFFF 60%); }
[data-testid="stChatMessage"] { background: #fff; border: 1px solid #ECEEF5; border-radius: 16px; padding: 10px 14px; }
div[data-testid="stTabs"] button p { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

pio.templates["ticket"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, sans-serif", color="#374151", size=12), paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)", title=dict(font=dict(size=15, color="#111827"), x=0, xanchor="left"),
    xaxis=dict(gridcolor="#EEF0F6", zerolinecolor="#E5E7EB"), yaxis=dict(gridcolor="#EEF0F6", zerolinecolor="#E5E7EB"),
    colorway=["#4F46E5", "#06B6D4", "#F59E0B", "#10B981", "#EC4899", "#8B5CF6", "#EF4444", "#94A3B8"],
    hoverlabel=dict(font_family="Inter, sans-serif"), margin=dict(l=10, r=10, t=50, b=10)))
pio.templates.default = "plotly_white+ticket"


def fig_show(fig, height=380, legend_bottom=False, container=st):
    fig.update_layout(height=height)
    if legend_bottom:
        fig.update_layout(legend=dict(orientation="h", y=-0.18, x=0, title=None))
    container.plotly_chart(fig, config={"displaylogo": False}, **W)


def hero(title, subtitle, tags=()):
    tag_html = "".join(f'<span class="tag">{t}</span>' for t in tags)
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p>{tag_html}</div>', unsafe_allow_html=True)


def kpi(col, label, value, sub="", color="#4F46E5"):
    col.markdown(f'<div class="kpi"><div class="bar" style="background:{color}"></div><div class="label">{label}</div>'
                 f'<div class="value">{value}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def section(title, note=""):
    st.markdown(f'<div class="section">{title}</div>' + (f'<div class="muted">{note}</div>' if note else ""),
                unsafe_allow_html=True)


def insight(text, color="#4F46E5"):
    st.markdown(f'<div class="insight" style="border-left-color:{color}">{text}</div>', unsafe_allow_html=True)


def rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def chip(cat):
    return f'<span class="chip" style="background:{COLORS.get(cat, "#64748B")}">{LABEL.get(cat, cat)}</span>'


# ------------------------------------------------------------------ data and models
@st.cache_data(show_spinner="Loading results...")
def load_data():
    tickets = pd.read_csv(P / "tickets_clean.csv", parse_dates=["created_at"])
    preds = pd.read_csv(P / "predictions.csv").drop(columns=["true_category"])
    clusters = pd.read_csv(P / "clusters.csv")
    clusters_bge = pd.read_csv(P / "clusters_bge.csv").rename(columns={"cluster_id": "bge_cluster", "theme": "bge_theme"})
    changes = pd.read_csv(P / "embedding_message_changes.csv")
    view = (tickets.merge(preds, on="ticket_id", how="left", validate="one_to_one")
            .merge(clusters[["ticket_id", "cluster_id", "theme"]], on="ticket_id", how="left", validate="one_to_one")
            .merge(clusters_bge, on="ticket_id", how="left", validate="one_to_one"))
    # out-of-fold BGE predictions are per message (every ticket with the same message gets the same prediction)
    bge = changes.set_index("message")
    view["bge_pred"] = view[T].map(bge["BGE + LR"])
    view["bge_conf"] = view[T].map(bge["BGE confidence"])
    view["bge_correct"] = view["bge_pred"] == view[Y]
    view["bge_review"] = view["bge_conf"] < config.CONFIDENCE_THRESHOLD
    view["bge_queue"] = np.where(view["bge_review"], config.REVIEW_QUEUE, view["bge_pred"].map(config.ROUTING))
    view["negative"] = view["customer_sentiment"].isin(["negative", "very_negative"])

    csv = ["model_comparison", "cv_scores", "classification_report", "confusion_matrix", "shap_global", "shap_local",
           "cluster_summary", "volume_anomalies", "hotspots", "ticket_anomalies", "anomaly_validation", "knowledge_base",
           "rag_retrieval_eval", "embedding_cv_scores", "embedding_per_class_f1", "embedding_class_report",
           "embedding_confidence_rule", "embedding_message_changes", "embedding_holdout_comparison", "shap_bge_global",
           "shap_bge_local", "shap_bge_deletion_test", "shap_overall_comparison", "shap_bge_examples",
           "cluster_summary_bge", "topic_clusters_bge", "clustering_bge_vs_lsa", "clustering_similarity",
           "clustering_bge_scan"]
    files = {name: pd.read_csv(P / f"{name}.csv") for name in csv}
    for name in ["metrics", "embedding_metrics", "shap_bge_checks"]:
        files[name] = json.load(open(P / f"{name}.json"))
    return view, files


@st.cache_resource(show_spinner="Loading the BGE language model (first time only)...")
def get_encoder():
    return load_encoder()


@st.cache_resource(show_spinner="Preparing the classifiers...")
def get_classifier():
    return TicketClassifier(get_encoder())


@st.cache_resource(show_spinner="Indexing the knowledge base for the assistant...")
def get_retriever():
    kb = pd.read_csv(P / "knowledge_base.csv")
    return rag.Retriever(kb, encoder=get_encoder())


try:
    DATA, F = load_data()
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}. Run notebooks 01 to 06 (and 02b, 03b, 04b) first.")
    st.stop()


def sidebar_filters():
    """Filters shown on the dashboard pages. Returns the filtered ticket table."""
    with st.sidebar:
        st.markdown("#### 🔎 Filters")
        dmin, dmax = DATA["created_at"].min().date(), DATA["created_at"].max().date()
        dates = st.date_input("Created between", (dmin, dmax), min_value=dmin, max_value=dmax, key="f_dates")
        if not (isinstance(dates, (list, tuple)) and len(dates) == 2):
            dates = (dmin, dmax)
        f_cat = st.multiselect("Category", CLASSES, format_func=LABEL.get, key="f_cat")
        f_pri = st.multiselect("Priority", PRIORITIES, key="f_pri")
        f_chan = st.multiselect("Channel", sorted(DATA["channel"].unique()), key="f_chan")
        f_prod = st.multiselect("Product area", sorted(DATA["product_area"].unique()), key="f_prod")
    m = DATA["created_at"].dt.date.between(dates[0], dates[1])
    for col, sel in [(Y, f_cat), ("priority", f_pri), ("channel", f_chan), ("product_area", f_prod)]:
        if sel:
            m &= DATA[col].isin(sel)
    v = DATA[m]
    st.sidebar.caption(f"**{len(v):,}** of {len(DATA):,} tickets in view")
    if v.empty:
        st.warning("No tickets match these filters.")
        st.stop()
    return v


# ================================================================== pages
def page_overview():
    v = sidebar_filters()
    em, m = F["embedding_metrics"], F["metrics"]
    hero("IT Support Ticket Intelligence",
         "Classify tickets, explain every prediction, find recurring issues and ask questions about the data.",
         [f"{len(DATA):,} tickets", "8 categories", f"Best model: BGE + LR · macro F1 {em['f1_macro']:.2f}",
          "Mistral-powered assistant"])

    res = v[v["status"] == "resolved"]
    c = st.columns(6)
    kpi(c[0], "Tickets in view", f"{len(v):,}", f"{v['created_at'].min():%b %Y} – {v['created_at'].max():%b %Y}")
    kpi(c[1], "Resolved", f"{v['status'].eq('resolved').mean():.0%}",
        f"{v['status'].isin(['open', 'in_progress', 'on_hold']).sum():,} still open", "#10B981")
    kpi(c[2], "Negative customers", f"{v['negative'].mean():.0%}", f"avg CSAT {v['csat_score'].mean():.2f} / 5", "#EF4444")
    kpi(c[3], "Median resolution", f"{res['resolution_time_hours'].median():.1f} h",
        f"reopen rate {v['reopened'].mean():.1%}", "#F59E0B")
    kpi(c[4], "Auto-routed (BGE)", f"{1 - v['bge_review'].mean():.0%}",
        f"<span class='up'>{v.loc[~v['bge_review'], 'bge_correct'].mean():.0%}</span> correct when auto-routed", "#4F46E5")
    kpi(c[5], "Model accuracy", f"{v['bge_correct'].mean():.0%}",
        f"BGE · TF-IDF was <span class='down'>{v['correct'].mean():.0%}</span>", "#8B5CF6")

    left, right = st.columns([1, 1.6])
    counts = v[Y].value_counts().rename_axis("category").reset_index(name="tickets")
    fig = px.pie(counts, names="category", values="tickets", hole=0.62, color="category", color_discrete_map=COLORS,
                 title="Ticket mix by category")
    fig.update_traces(textinfo="percent", hovertemplate="%{label}: %{value:,} tickets<extra></extra>")
    fig.add_annotation(text=f"<b>{len(v):,}</b><br>tickets", showarrow=False, font=dict(size=16))
    fig_show(fig, 380, container=left)

    monthly = v.groupby([pd.Grouper(key="created_at", freq="MS"), Y]).size().rename("tickets").reset_index()
    fig = px.area(monthly, x="created_at", y="tickets", color=Y, color_discrete_map=COLORS,
                  title="Tickets per month by category")
    fig.update_layout(xaxis_title=None, yaxis_title=None, legend_title=None)
    fig_show(fig, 380, legend_bottom=True, container=right)

    left, right = st.columns(2)
    sat = v.groupby(Y).agg(negative=("negative", "mean"), csat=("csat_score", "mean")).reset_index().sort_values("negative")
    fig = px.bar(sat, x="negative", y=Y, orientation="h", color=Y, color_discrete_map=COLORS, text="negative",
                 hover_data={"csat": ":.2f"}, title="Share of negative customers by category")
    fig.update_traces(texttemplate="%{x:.0%}", textposition="outside")
    fig.add_vline(x=v["negative"].mean(), line_dash="dot", line_color="#6B7280",
                  annotation_text=f"overall {v['negative'].mean():.0%}", annotation_position="top")
    fig.update_layout(showlegend=False, xaxis_tickformat=".0%", xaxis_title=None, yaxis_title=None,
                      xaxis_range=[0, max(0.8, sat["negative"].max() * 1.2)])
    fig_show(fig, 380, container=left)

    pr = res.groupby("priority")["resolution_time_hours"].describe(percentiles=[.25, .5, .75]).reindex(PRIORITIES).reset_index()
    fig = go.Figure()
    for col, name, color in [("25%", "25th percentile", "#C7D2FE"), ("50%", "median", "#4F46E5"), ("75%", "75th percentile", "#A5B4FC")]:
        fig.add_bar(x=pr["priority"], y=pr[col], name=name, marker_color=color,
                    text=pr[col].round(1), textposition="outside")
    fig.update_layout(title="Resolution time by priority (hours)", barmode="group", yaxis_title=None)
    fig_show(fig, 380, legend_bottom=True, container=right)

    left, right = st.columns(2)
    queues = v["bge_queue"].value_counts().rename_axis("queue").reset_index(name="tickets")
    fig = px.bar(queues, x="tickets", y="queue", orientation="h", text="tickets",
                 title="Where tickets are routed (BGE model, 60% confidence rule)")
    fig.update_traces(marker_color=["#F97316" if q == config.REVIEW_QUEUE else "#4F46E5" for q in queues["queue"]],
                      texttemplate="%{x:,}", textposition="outside")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title=None, yaxis_title=None)
    fig_show(fig, 380, container=left)
    left.caption("Orange = manual review. The category → team routing table is an assumption (the data has no assignment group).")

    heat = pd.crosstab(v[Y], v["priority"]).reindex(columns=PRIORITIES)
    fig = px.imshow(heat, text_auto=",", color_continuous_scale="Purples", aspect="auto",
                    title="Tickets by category and priority", labels=dict(x="", y="", color="tickets"))
    fig.update_coloraxes(showscale=False)
    fig_show(fig, 380, container=right)

    section("Key insights")
    worst = sat.iloc[-1]
    c = st.columns(3)
    with c[0]:
        insight(f"😟 <b>{LABEL[worst[Y]]}</b> has the unhappiest customers: <b>{worst['negative']:.0%}</b> negative vs "
                f"{v['negative'].mean():.0%} overall. Problem categories sit around 60%, requests around 20%.", "#EF4444")
    with c[1]:
        insight(f"⚡ Priority drives resolution time: median <b>{pr.loc[pr['priority'] == 'urgent', '50%'].iloc[0]:.1f} h</b> "
                f"for urgent vs <b>{pr.loc[pr['priority'] == 'low', '50%'].iloc[0]:.1f} h</b> for low. "
                "The category barely changes it.", "#F59E0B")
    with c[2]:
        rule = F["embedding_confidence_rule"].set_index("model")
        insight(f"🤖 The BGE model sends only <b>{rule.loc['BGE + LR', 'sent to review']:.0%}</b> of tickets to manual review "
                f"(TF-IDF: {rule.loc['TF-IDF + LR', 'sent to review']:.0%}) and auto-routed tickets are "
                f"<b>{rule.loc['BGE + LR', 'auto-routed accuracy']:.0%}</b> correct.", "#4F46E5")


# ------------------------------------------------------------------ classify
EXAMPLES = {
    "🔐 Locked out": "I forgot my password and now my account is locked after too many attempts",
    "💳 Double charge": "Please refund the extra charge on my credit card, I was billed twice this month",
    "🐞 App crash": "The mobile app freezes every time I upload a photo",
    "🛡️ Suspicious login": "Someone logged into my account from another country last night",
    "🐢 Slow reports": "The analytics dashboard takes forever to load our weekly report",
    "☕ Out of scope": "The coffee machine on the third floor is broken",
}


def set_ticket(text):
    st.session_state["ticket_text"] = text


def outlook(category, priority, sla, product):
    """Historical outcomes for tickets like this one (same category / priority / SLA / product)."""
    d = DATA
    res = d[d["status"] == "resolved"]
    same = res[(res[Y] == category) & (res["priority"] == priority)]
    sla_same = res[(res["priority"] == priority) & (res["sla_plan"] == sla)]
    cat_prod = d[(d[Y] == category) & (d["product_area"] == product)]
    return {
        "median_h": same["resolution_time_hours"].median(), "p75_h": same["resolution_time_hours"].quantile(.75),
        "n_same": len(same), "sla_median_h": sla_same["resolution_time_hours"].median(),
        "reopen": d[(d[Y] == category) & (d["priority"] == priority)]["reopened"].mean(),
        "negative": cat_prod["negative"].mean(), "csat": cat_prod["csat_score"].mean(), "n_prod": len(cat_prod),
    }


def gauge(conf, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=conf * 100, number={"suffix": "%", "font": {"size": 34, "color": "#111827"}},
        gauge={"axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "#fff"}, "bar": {"color": color, "thickness": 0.75},
               "bgcolor": "#F3F4F6", "borderwidth": 0,
               "threshold": {"line": {"color": "#F97316", "width": 3}, "thickness": 0.9,
                             "value": config.CONFIDENCE_THRESHOLD * 100}}))
    fig.update_layout(height=210, margin=dict(l=20, r=20, t=30, b=0), title=dict(text="Confidence", x=0.5, xanchor="center"))
    return fig


def page_classify():
    hero("🔮 Classify a ticket",
         "Type a ticket description. The model predicts its category, routes it, explains which words drove the decision "
         "and shows what usually happens to tickets like it.", ["BGE + LR (best) or TF-IDF + LR", "Word-level SHAP",
                                                                 "60% confidence rule"])
    clf = get_classifier()
    single, batch = st.tabs(["✍️  Single ticket", "📦  Batch (CSV upload)"])

    with single:
        st.markdown("**Try an example**")
        cols = st.columns(len(EXAMPLES))
        for col, (name, text) in zip(cols, EXAMPLES.items()):
            col.button(name, on_click=set_ticket, args=(text,), **W)

        with st.form("classify"):
            text = st.text_area("Ticket description", key="ticket_text", height=110,
                                placeholder="e.g. I can't log in, it says my password is incorrect")
            with st.expander("🧾 Ticket details (optional): used for the operational outlook", expanded=False):
                st.caption("The category is predicted from the description only: in this data, fields such as priority or "
                           "channel carry no information about the category. They are used to estimate resolution time, "
                           "reopen risk and customer sentiment.")
                c = st.columns(4)
                priority = c[0].selectbox("Priority", PRIORITIES, index=2)
                sla = c[1].selectbox("SLA plan", sorted(DATA["sla_plan"].unique()), index=1)
                channel = c[2].selectbox("Channel", sorted(DATA["channel"].unique()))
                product = c[3].selectbox("Product area", sorted(DATA["product_area"].unique()))
                c = st.columns(3)
                segment = c[0].selectbox("Customer segment", sorted(DATA["customer_segment"].unique()))
                platform = c[1].selectbox("Platform", sorted(DATA["platform"].unique()))
                region = c[2].selectbox("Region", sorted(DATA["region"].unique()))
            options = ["BGE + LR (best)", "TF-IDF + LR", "Compare both"] if clf.has_bge else ["TF-IDF + LR"]
            choice = st.segmented_control("Model", options, default=options[0], key="model_choice")
            submitted = st.form_submit_button("✨ Classify ticket", type="primary", **W)

        if submitted:
            if not text.strip():
                st.warning("Please type a ticket description first.")
            else:
                with st.spinner("Classifying and explaining..."):
                    main = "tfidf" if (choice or options[0]).startswith("TF-IDF") else "bge"
                    probs = {m: clf.predict(text, m) for m in (["bge", "tfidf"] if clf.has_bge else ["tfidf"])}
                    label = probs[main].index[0]
                    shap_df, method = clf.explain_bge(text, label) if main == "bge" else clf.explain_tfidf(text, label)
                    st.session_state["result"] = dict(
                        text=text, main=main, compare=(choice == "Compare both"), probs=probs, shap=shap_df,
                        method=method, similar=clf.similar(text, 5),
                        fields=dict(priority=priority, sla=sla, channel=channel, product=product, segment=segment,
                                    platform=platform, region=region), ai=None)

        r = st.session_state.get("result")
        if r:
            render_result(r)

    with batch:
        render_batch(clf)


def render_result(r):
    name = {"bge": "BGE + LR", "tfidf": "TF-IDF + LR"}
    p = r["probs"][r["main"]]
    label, conf = p.index[0], p.iloc[0]
    dec = TicketClassifier.decision(label, conf)
    color = COLORS[label]
    st.markdown("")
    left, mid, right = st.columns([1.35, 1, 1.15])
    with left:
        status = ("🟠 Manual review: confidence below 60%" if dec["review"] else f"🟢 Auto-route to {dec['queue']}")
        st.markdown(f"""<div class="pred" style="background: linear-gradient(135deg, {color} 0%, #111827 140%);">
            <div class="small">Predicted category · {name[r['main']]}</div>
            <div class="big">{LABEL[label]}</div>
            <span class="pill">{status}</span><br>
            <span class="pill">Runner-up: {LABEL[p.index[1]]} ({p.iloc[1]:.0%})</span>
            <span class="pill">Team: {config.ROUTING[label]}</span></div>""", unsafe_allow_html=True)
        if dec["review"]:
            st.info("The model isn't sure, so a person should confirm the category. This is how out-of-scope or "
                    "ambiguous tickets are caught.")
    with mid:
        st.plotly_chart(gauge(conf, color), config={"displayModeBar": False}, **W)
    with right:
        top = p.head(5).rename_axis("category").reset_index(name="probability").iloc[::-1]
        fig = px.bar(top, x="probability", y="category", orientation="h", color="category", color_discrete_map=COLORS,
                     text="probability", title="Top 5 categories")
        fig.update_traces(texttemplate="%{x:.0%}", textposition="outside")
        fig.update_layout(showlegend=False, xaxis_tickformat=".0%", xaxis_range=[0, 1.15], xaxis_title=None, yaxis_title=None)
        fig_show(fig, 250)

    why, compare, similar, look, ai = st.tabs(["🧠 Why this prediction", "⚖️ Model comparison", "🔍 Similar past tickets",
                                               "📈 Operational outlook", "✨ AI explanation"])
    with why:
        s = r["shap"]
        st.markdown(f"**Word contributions towards _{LABEL[label]}_** · SHAP, {r['method']}. "
                    "<span style='color:#4F46E5'>■ pushes towards</span> · <span style='color:#EF4444'>■ pushes against</span>",
                    unsafe_allow_html=True)
        if r["main"] == "bge":
            st.markdown(f'<div class="textbox">{highlight_html(s)}</div>', unsafe_allow_html=True)
        if s.empty:
            st.info("None of these words were seen in training, so there is nothing to explain: the ticket is likely out of scope.")
        else:
            d = s.reindex(s["shap"].abs().sort_values(ascending=False).index).head(10).iloc[::-1]
            fig = px.bar(d, x="shap", y="word", orientation="h", text="shap")
            fig.update_traces(marker_color=np.where(d["shap"] > 0, "#4F46E5", "#EF4444"), texttemplate="%{x:+.2f}")
            fig.update_layout(xaxis_title="SHAP value (contribution to the model score)", yaxis_title=None)
            fig_show(fig, 360)
    with compare:
        if len(r["probs"]) < 2:
            st.info("Install sentence-transformers to compare with the BGE model.")
        else:
            both = pd.DataFrame({name[m]: r["probs"][m] for m in r["probs"]}).reindex(CLASSES)
            long = both.rename_axis("category").reset_index().melt("category", var_name="model", value_name="probability")
            fig = px.bar(long, x="category", y="probability", color="model", barmode="group",
                         color_discrete_map=MODEL_COLORS, title="Probability per category, both models")
            fig.update_layout(yaxis_tickformat=".0%", xaxis_title=None, yaxis_title=None)
            fig_show(fig, 360, legend_bottom=True)
            c = st.columns(2)
            for col, m in zip(c, ["bge", "tfidf"]):
                pm = r["probs"][m]
                dm = TicketClassifier.decision(pm.index[0], pm.iloc[0])
                kpi(col, name[m], LABEL[pm.index[0]], f"{pm.iloc[0]:.0%} confidence · "
                    f"{'manual review' if dm['review'] else 'auto-route to ' + dm['queue']}", MODEL_COLORS[name[m]])
            st.caption("TF-IDF matches words it has seen; BGE understands meaning, so it handles new wording much better "
                       "(macro F1 0.82 vs 0.54 on unseen sentences).")
    with similar:
        sim = r["similar"].copy()
        st.markdown("The closest descriptions among the 96 known ticket sentences, with how many tickets used them.")
        sim["category"] = sim[Y].map(LABEL)
        cols = ["similarity", T, "category", "tickets"] + [c for c in ["theme", "topic_words"] if c in sim]
        st.dataframe(sim[cols], hide_index=True, **W, column_config={
            "similarity": st.column_config.ProgressColumn("Similarity", min_value=0, max_value=1, format="%.2f"),
            T: st.column_config.TextColumn("Known description", width="large"),
            "tickets": st.column_config.NumberColumn("Tickets", format="%d"),
            "theme": "Recurring theme", "topic_words": "Topic"})
    with look:
        f = r["fields"]
        o = outlook(label, f["priority"], f["sla"], f["product"])
        st.markdown(f"Historical outcomes for **{LABEL[label]}** tickets with **{f['priority']}** priority, "
                    f"**{f['sla']}** SLA and product area **{f['product']}**.")
        c = st.columns(4)
        kpi(c[0], "Expected resolution", f"{o['median_h']:.1f} h", f"75% within {o['p75_h']:.1f} h · {o['n_same']:,} past tickets",
            "#4F46E5")
        kpi(c[1], f"{f['sla'].capitalize()} SLA median", f"{o['sla_median_h']:.1f} h", f"all {f['priority']}-priority tickets",
            "#06B6D4")
        kpi(c[2], "Reopen risk", f"{o['reopen']:.1%}", f"vs {DATA['reopened'].mean():.1%} overall", "#F59E0B")
        kpi(c[3], "Unhappy customers", f"{o['negative']:.0%}", f"avg CSAT {o['csat']:.2f} · {o['n_prod']:,} past tickets",
            "#EF4444")
        st.caption("Descriptive statistics from the 20,000 historical tickets, not a prediction model.")
    with ai:
        provider = rag.llm_provider()
        st.caption(f"Uses {'Mistral' if provider == 'mistral' else provider} with the SHAP words and the retrieved facts "
                   "about this category.")
        if st.button("✨ Explain this prediction in plain English", type="primary"):
            with st.spinner("Asking the assistant..."):
                terms = list(zip(r["shap"]["word"], r["shap"]["shap"]))
                terms.sort(key=lambda t: -abs(t[1]))
                r["ai"] = rag.explain_prediction(r["text"], label, conf, terms, get_retriever(), model_name=name[r["main"]])
        if r.get("ai"):
            text, mode = r["ai"]
            st.success(f"**{'AI-generated' if mode == 'LLM' else 'Auto-generated (offline)'}**\n\n{text}")


def render_batch(clf):
    st.markdown("Upload a CSV with one ticket per row. You'll get the category, confidence, routing decision and "
                "probabilities for every row.")
    sample = DATA[[T]].drop_duplicates().sample(12, random_state=1).rename(columns={T: "description"})
    st.download_button("⬇️ Download a sample CSV", sample.to_csv(index=False), "sample_tickets.csv", "text/csv")
    up = st.file_uploader("CSV file", type=["csv"])
    if not up:
        return
    df = pd.read_csv(up)
    guess = next((c for c in df.columns if any(k in c.lower() for k in ["desc", "message", "text", "ticket"])), df.columns[0])
    c = st.columns(2)
    col = c[0].selectbox("Column with the ticket text", list(df.columns), index=list(df.columns).index(guess))
    model = c[1].segmented_control("Model", ["BGE + LR", "TF-IDF + LR"] if clf.has_bge else ["TF-IDF + LR"],
                                   default="BGE + LR" if clf.has_bge else "TF-IDF + LR", key="batch_model")
    with st.spinner(f"Classifying {len(df):,} tickets..."):
        proba = clf.predict_many(df[col].fillna(""), "bge" if model == "BGE + LR" else "tfidf")
    out = df.copy()
    out["predicted_category"] = proba.idxmax(axis=1)
    out["confidence"] = proba.max(axis=1)
    out["decision"] = np.where(out["confidence"] < config.CONFIDENCE_THRESHOLD, "manual review", "auto-route")
    out["queue"] = np.where(out["decision"] == "manual review", config.REVIEW_QUEUE,
                            out["predicted_category"].map(config.ROUTING))
    c = st.columns(3)
    kpi(c[0], "Tickets", f"{len(out):,}")
    kpi(c[1], "Auto-routed", f"{(out['decision'] == 'auto-route').mean():.0%}", color="#10B981")
    kpi(c[2], "Manual review", f"{(out['decision'] == 'manual review').mean():.0%}", color="#F97316")
    left, right = st.columns([1.6, 1])
    left.dataframe(out, hide_index=True, **W, column_config={
        "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f")})
    fig = px.histogram(out, y="predicted_category", color="predicted_category", color_discrete_map=COLORS,
                       title="Predicted categories")
    fig.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"}, xaxis_title=None, yaxis_title=None)
    fig_show(fig, 360, container=right)
    st.download_button("⬇️ Download predictions", pd.concat([out, proba.add_prefix("proba_")], axis=1).to_csv(index=False),
                       "ticket_predictions.csv", "text/csv", type="primary")


# ------------------------------------------------------------------ model performance
def threshold_curve(conf, correct, thresholds):
    rows = []
    for t in thresholds:
        auto = conf >= t
        rows.append({"threshold": t, "sent to review": 1 - auto.mean(),
                     "auto-routed accuracy": correct[auto].mean() if auto.any() else np.nan})
    return pd.DataFrame(rows)


def page_models():
    em, m = F["embedding_metrics"], F["metrics"]
    hero("📈 Model performance",
         "Every number is measured on wording the model has never seen (grouped cross-validation by message), "
         "so it reflects how the models handle new tickets.", ["4 text representations", "5 classifiers",
                                                                "out-of-fold on 20,000 tickets"])
    c = st.columns(5)
    for col, (lab, key_b, key_t, color) in zip(c, [
            ("Macro F1", "f1_macro", "macro_f1", "#4F46E5"), ("Accuracy", "accuracy", "accuracy", "#10B981"),
            ("Macro precision", "precision_macro", "macro_precision", "#06B6D4"),
            ("Macro recall", "recall_macro", "macro_recall", "#F59E0B"),
            ("ROC-AUC (macro)", "roc_auc_macro", "roc_auc_macro", "#8B5CF6")]):
        kpi(col, lab, f"{em[key_b]:.3f}", f"BGE + LR · TF-IDF + LR: {m[key_t]:.3f} "
            f"<span class='up'>(+{em[key_b] - m[key_t]:.2f})</span>", color)

    tab_board, tab_class, tab_conf, tab_thr, tab_err = st.tabs(
        ["🏆 Leaderboard", "🎯 Per-class scores", "🧩 Confusion matrix", "🎚️ Confidence threshold", "🔬 Error analysis"])

    with tab_board:
        cv = F["embedding_cv_scores"]
        reps = ["TF-IDF", "MiniLM", "MPNet", "BGE"]
        clfs = ["Logistic Regression", "Linear SVM", "kNN (k=3)", "Random Forest", "Naive Bayes"]
        piv = cv.pivot(index="classifier", columns="representation", values="f1_macro_mean").reindex(index=clfs, columns=reps)
        left, right = st.columns([1.2, 1])
        fig = px.imshow(piv, text_auto=".2f", color_continuous_scale="Purples", aspect="auto", zmin=0.4, zmax=0.85,
                        title="Macro F1 (grouped 5-fold CV): text representation × classifier",
                        labels=dict(x="", y="", color="macro F1"))
        fig.update_coloraxes(showscale=False)
        fig_show(fig, 400, container=left)
        best = cv.sort_values("f1_macro_mean", ascending=False).head(8).copy()
        best["model"] = best["representation"] + " + " + best["classifier"]
        fig = px.bar(best.iloc[::-1], x="f1_macro_mean", y="model", orientation="h", error_x="f1_macro_std",
                     text="f1_macro_mean", title="Top 8 combinations (± std over folds)")
        fig.update_traces(marker_color=["#4F46E5" if r == "BGE" else "#A5B4FC" for r in best.iloc[::-1]["representation"]],
                          texttemplate="%{x:.3f}", textposition="inside", insidetextanchor="start",
                          textfont_color="white")
        fig.update_layout(xaxis_range=[0, 1], xaxis_title=None, yaxis_title=None)
        fig_show(fig, 400, container=right)
        c = st.columns(3)
        with c[0]:
            insight("🏅 <b>BGE + Linear SVM / Logistic Regression</b> are best (macro F1 ≈ 0.82). Logistic Regression is used "
                    "because it gives probabilities for the confidence rule.")
        with c[1]:
            insight("📚 A better embedding model helps a lot: MiniLM 0.71 → MPNet 0.79 → BGE 0.81 (with LR). "
                    "Every TF-IDF model is stuck at 0.59.", "#06B6D4")
        with c[2]:
            insight("🌲 Random Forest is weakest on embeddings: meaning is spread over hundreds of dimensions and there "
                    "are only ~77 training sentences per fold.", "#F59E0B")
        with st.expander("Why not a random train/test split? (data leakage)"):
            st.markdown("The 20,000 tickets use only **96 unique sentences**. A random split puts the same sentences in "
                        "train and test and gives a misleading **macro F1 of 1.00**. Every score here comes from a split "
                        "**by message**, so the test only contains wording the model has never seen.")
            st.dataframe(F["embedding_holdout_comparison"].round(3), hide_index=True, **W)

    with tab_class:
        pc = F["embedding_per_class_f1"].rename(columns={"Unnamed: 0": "category"})
        pc = pc[pc["category"].isin(CLASSES)]
        long = pc.melt("category", var_name="model", value_name="F1")
        fig = px.bar(long, x="category", y="F1", color="model", barmode="group", title="F1 per category (out-of-fold)",
                     color_discrete_sequence=["#D4D4D8", "#A5B4FC", "#818CF8", "#4F46E5"])
        fig.update_layout(yaxis_range=[0, 1.05], xaxis_title=None)
        fig_show(fig, 420, legend_bottom=True)
        rep = F["embedding_class_report"].rename(columns={"Unnamed: 0": "category"})
        rep = rep[rep["category"].isin(CLASSES)][["category", "precision", "recall", "f1-score", "roc_auc", "support"]]
        st.markdown("**BGE + LR per-class report**")
        st.dataframe(rep, hide_index=True, **W, column_config={
            c: st.column_config.ProgressColumn(c.capitalize(), min_value=0, max_value=1, format="%.2f")
            for c in ["precision", "recall", "f1-score", "roc_auc"]} | {"support": st.column_config.NumberColumn("Tickets")})
        insight("The two classes TF-IDF missed completely improve most with BGE: <b>account access 0.00 → 0.80</b> and "
                "<b>other 0.00 → 0.49</b>. 'Other' stays hardest because its sentences are vague by design.")

    with tab_conf:
        c = st.columns([1, 1, 2])
        which = c[0].segmented_control("Model", ["BGE + LR", "TF-IDF + LR"], default="BGE + LR", key="cm_model")
        norm = c[1].toggle("Show as % of each actual category", value=True)
        if which == "TF-IDF + LR":
            cm = F["confusion_matrix"].set_index("Unnamed: 0").reindex(index=CLASSES, columns=CLASSES)
        else:
            ch = F["embedding_message_changes"]
            cm = ch.pivot_table(index="true", columns="BGE + LR", values="tickets", aggfunc="sum", fill_value=0) \
                   .reindex(index=CLASSES, columns=CLASSES, fill_value=0)
        shown = cm.div(cm.sum(axis=1).replace(0, 1), axis=0) if norm else cm
        fig = px.imshow(shown, text_auto=".0%" if norm else ",", color_continuous_scale="Blues", aspect="auto",
                        labels=dict(x="Predicted", y="Actual", color=""), title=f"Confusion matrix · {which or 'BGE + LR'}")
        fig.update_coloraxes(showscale=False)
        fig_show(fig, 520)

    with tab_thr:
        thresholds = np.round(np.arange(0.30, 0.96, 0.05), 2)
        curves = pd.concat([
            threshold_curve(DATA["bge_conf"].values, DATA["bge_correct"].values, thresholds).assign(model="BGE + LR"),
            threshold_curve(DATA["confidence"].values, DATA["correct"].values, thresholds).assign(model="TF-IDF + LR")])
        t = st.slider("Manual-review threshold: tickets below this confidence go to a person", 0.30, 0.95,
                      config.CONFIDENCE_THRESHOLD, 0.05)
        c = st.columns(4)
        for col, (mname, conf_col, corr_col) in zip([c[:2], c[2:]], [("BGE + LR", "bge_conf", "bge_correct"),
                                                                    ("TF-IDF + LR", "confidence", "correct")]):
            auto = DATA[conf_col] >= t
            kpi(col[0], f"{mname}: auto-routed", f"{auto.mean():.0%}",
                f"{DATA.loc[auto, corr_col].mean():.0%} correct" if auto.any() else "", MODEL_COLORS[mname])
            kpi(col[1], f"{mname}: to review", f"{1 - auto.mean():.0%}",
                f"{DATA.loc[~auto, corr_col].mean():.0%} correct (mostly mistakes caught)" if (~auto).any() else "", "#F97316")
        left, right = st.columns(2)
        for col, metric, title in [(left, "sent to review", "Share of tickets sent to manual review"),
                                   (right, "auto-routed accuracy", "Accuracy of auto-routed tickets")]:
            fig = px.line(curves, x="threshold", y=metric, color="model", markers=True, color_discrete_map=MODEL_COLORS,
                          title=title)
            fig.add_vline(x=t, line_dash="dash", line_color="#F97316")
            fig.update_layout(yaxis_tickformat=".0%", yaxis_title=None)
            fig_show(fig, 340, legend_bottom=True, container=col)

    with tab_err:
        ch = F["embedding_message_changes"]
        em2 = F["embedding_metrics"]
        c = st.columns(4)
        kpi(c[0], "Sentences right · TF-IDF", f"{em2['messages_right_tfidf']} / 96", color="#A1A1AA")
        kpi(c[1], "Sentences right · BGE", f"{em2['messages_right_bge']} / 96", color="#4F46E5")
        kpi(c[2], "Fixed by BGE", f"{em2['messages_fixed']}", color="#10B981")
        kpi(c[3], "Broken by BGE", f"{em2['messages_broken']}", color="#EF4444")
        wrong = ch[~ch["TF-IDF ok"] | ~ch["BGE ok"]].copy()
        wrong["status"] = np.where(wrong["BGE ok"], "✅ fixed by BGE", "❌ still wrong")
        st.dataframe(wrong[["status", "message", "true", "TF-IDF + LR", "BGE + LR", "BGE confidence", "tickets"]]
                     .sort_values("status"), hide_index=True, **W, column_config={
                         "message": st.column_config.TextColumn("Sentence", width="large"),
                         "BGE confidence": st.column_config.ProgressColumn("BGE confidence", min_value=0, max_value=1,
                                                                           format="%.2f")})
        st.caption("3 of the 4 sentences BGE still gets wrong have confidence below 60%, so they go to manual review.")


# ------------------------------------------------------------------ explainability
def page_explain():
    hero("🧠 Explainability (SHAP)",
         "Which words push a ticket into each category? SHAP gives every word a score for how much it moved the prediction.",
         ["TF-IDF: exact linear SHAP", "BGE: exact word-level Shapley values", "faithfulness checked"])
    which = st.segmented_control("Model", ["BGE + LR", "TF-IDF + LR"], default="BGE + LR", key="shap_model") or "BGE + LR"
    g = F["shap_bge_global"] if which == "BGE + LR" else F["shap_global"]

    section("What drives each category", "Mean SHAP value of each word over the category's tickets.")
    cat = st.pills("Category", CLASSES, default="account_access", format_func=LABEL.get, key="shap_cat") or "account_access"
    d = g[g["category"] == cat].sort_values("mean_shap").tail(10)
    left, right = st.columns([1.3, 1])
    fig = px.bar(d, x="mean_shap", y="term", orientation="h", text="mean_shap",
                 title=f"Top words pushing tickets into “{LABEL[cat]}” · {which}")
    fig.update_traces(marker_color=COLORS[cat], texttemplate="%{x:.2f}")
    fig.update_layout(xaxis_title="mean SHAP value", yaxis_title=None)
    fig_show(fig, 400, container=left)
    with right:
        other = F["shap_global"] if which == "BGE + LR" else F["shap_bge_global"]
        other_name = "TF-IDF + LR" if which == "BGE + LR" else "BGE + LR"
        st.markdown(f"**Same category, {other_name}**")
        o = other[other["category"] == cat].sort_values("mean_shap", ascending=False).head(10)
        st.dataframe(o[["term", "mean_shap"]], hide_index=True, **W, column_config={
            "mean_shap": st.column_config.ProgressColumn("mean SHAP", min_value=0, max_value=float(o["mean_shap"].max()),
                                                         format="%.2f")})

    section("Most influential words overall", "Orange = stop words (common words such as 'my', 'the', 'how').")
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    so = F["shap_overall_comparison"]
    c = st.columns(2)
    for col, mname in zip(c, ["TF-IDF + LR", "BGE + LR"]):
        d = so[so["model"] == mname].sort_values("mean_abs_shap").copy()
        stop = d["term"].apply(lambda t: all(w in ENGLISH_STOP_WORDS for w in str(t).split()))
        fig = px.bar(d, x="mean_abs_shap", y="term", orientation="h", title=f"{mname} · {stop.sum()} of top 20 are stop words")
        fig.update_traces(marker_color=np.where(stop, "#FB923C", MODEL_COLORS[mname]))
        fig.update_layout(xaxis_title="mean |SHAP|", yaxis_title=None)
        fig_show(fig, 520, container=col)
    chk = F["shap_bge_checks"]
    insight(f"Both models spend a similar share of their explanation on stop words (TF-IDF {chk['stop_word_share_tfidf']:.0%}, "
            f"BGE {chk['stop_word_share_bge']:.0%}). The difference is which words carry the most weight: TF-IDF's top list is "
            "dominated by word pairs from one-off sentences, while BGE's is mostly problem words (error, suspicious, password, "
            "crashing, slow).")

    left, right = st.columns([1, 1])
    with left:
        section("Can we trust the BGE explanations?", "Deletion test: delete the top SHAP words vs random words.")
        dt = F["shap_bge_deletion_test"]
        long = dt.melt("words deleted", ["top SHAP words", "random words"], var_name="deleted", value_name="drop")
        fig = px.bar(long, x="words deleted", y="drop", color="deleted", barmode="group", text="drop",
                     color_discrete_map={"top SHAP words": "#4F46E5", "random words": "#D4D4D8"},
                     title="Drop in confidence of the original prediction")
        fig.update_traces(texttemplate="%{y:.0%}")
        fig.update_layout(yaxis_tickformat=".0%", yaxis_title=None, xaxis=dict(tickmode="array", tickvals=[1, 2, 3]))
        fig_show(fig, 360, legend_bottom=True)
        st.caption(f"SHAP values add up to the model score (max error {chk['additivity_max_error']:.0e}) and match the shap "
                   f"library within {chk['shap_library_max_difference']:.0e}.")
    with right:
        section("Explain a known description")
        loc = F["shap_bge_local"] if which == "BGE + LR" else F["shap_local"]
        msg = st.selectbox("Description", sorted(loc[T].unique()), key="shap_msg")
        d = loc[loc[T] == msg]
        st.markdown(f"Predicted {chip(d['predicted'].iloc[0])} with **{d['confidence'].iloc[0]:.0%}** confidence",
                    unsafe_allow_html=True)
        if which == "BGE + LR":
            order = {w: v for w, v in zip(d["term"], d["shap"])}
            words = pd.DataFrame({"word": msg.split()})
            words["shap"] = words["word"].map(order).fillna(0.0)
            st.markdown(f'<div class="textbox">{highlight_html(words)}</div>', unsafe_allow_html=True)
        d = d.sort_values("shap")
        fig = px.bar(d, x="shap", y="term", orientation="h")
        fig.update_traces(marker_color=np.where(d["shap"] > 0, "#4F46E5", "#EF4444"))
        fig.update_layout(xaxis_title="SHAP value", yaxis_title=None, title="Word contributions")
        fig_show(fig, 300)


# ------------------------------------------------------------------ recurring issues
def page_themes():
    v = sidebar_filters()
    hero("🔁 Recurring issues",
         "Tickets with similar descriptions are grouped into themes (within a category) and topics (across categories).",
         ["K-Means + silhouette", "BGE embeddings vs TF-IDF + LSA", "cross-category topics"])
    which = st.segmented_control("Clustering", ["BGE embeddings (4b)", "TF-IDF + LSA (04)"], default="BGE embeddings (4b)",
                                 key="theme_model") or "BGE embeddings (4b)"
    bge = which.startswith("BGE")
    summ = F["cluster_summary_bge"] if bge else F["cluster_summary"]
    id_col = "bge_cluster" if bge else "cluster_id"
    counts = v[id_col].value_counts().rename("tickets_in_view")
    th = summ.merge(counts, left_on="cluster_id", right_index=True, how="left").fillna({"tickets_in_view": 0})
    th["label"] = th["theme"].str.split(", ").str[:2].str.join(", ")
    th["category_label"] = th["category"].map(LABEL)

    c = st.columns(4)
    kpi(c[0], "Themes", f"{len(th)}", "within the 8 categories")
    kpi(c[1], "Largest theme", f"{int(th['tickets_in_view'].max()):,}", th.loc[th["tickets_in_view"].idxmax(), "label"], "#10B981")
    kpi(c[2], "Avg cohesion", f"{th['cohesion'].mean():.2f}", "1 = all sentences say the same", "#06B6D4")
    agree = F["clustering_bge_vs_lsa"]
    kpi(c[3], "Same themes as notebook 04", f"{(agree.iloc[:, -1] == 1).sum()} / 8", "categories with identical grouping",
        "#8B5CF6")

    left, right = st.columns([1.5, 1])
    tm = th[th["tickets_in_view"] > 0]
    fig = px.treemap(tm, path=[px.Constant("All tickets"), "category_label", "label"], values="tickets_in_view",
                     color="category", color_discrete_map=COLORS | {"(?)": "#E5E7EB"},
                     title="Themes by category (size = tickets in view)", hover_data={"cohesion": ":.2f"})
    fig.update_traces(root_color="#F3F4F6", marker=dict(cornerradius=6), textinfo="label+value")
    fig_show(fig, 520, container=left)
    with right:
        pick = st.selectbox("Look at a theme", th.sort_values("tickets_in_view", ascending=False)["cluster_id"],
                            format_func=lambda c_: f"{LABEL[th.set_index('cluster_id').loc[c_, 'category']]}: "
                                                   f"{th.set_index('cluster_id').loc[c_, 'label']}")
        row = th.set_index("cluster_id").loc[pick]
        st.markdown(f"{chip(row['category'])} &nbsp; **{row['theme']}**", unsafe_allow_html=True)
        cc = st.columns(3)
        cc[0].metric("Tickets", f"{int(row['tickets_in_view']):,}")
        cc[1].metric("Descriptions", int(row["unique_sentences"]))
        cc[2].metric("Cohesion", f"{row['cohesion']:.2f}")
        st.markdown("**Representative descriptions**")
        for text_ in v.loc[v[id_col] == pick, T].value_counts().head(5).index:
            st.markdown(f"- {text_}")
        tr = v[v[id_col] == pick].groupby(pd.Grouper(key="created_at", freq="QS")).size().rename("tickets").reset_index()
        fig = px.area(tr, x="created_at", y="tickets", title="Theme volume per quarter")
        fig.update_traces(line_color=COLORS[row["category"]])
        fig.update_layout(xaxis_title=None, yaxis_title=None, yaxis_rangemode="tozero")
        fig_show(fig, 220)

    section("Topics that cut across categories (BGE)",
            "With 8 clusters over all tickets, BGE groups by subject and intent, e.g. everything about login or billing.")
    topics = F["topic_clusters_bge"].copy()
    topics["name"] = "T" + topics["topic"].astype(str) + ": " + topics["top words"].str.split(", ").str[:2].str.join(", ")
    flows = v.groupby([Y, "topic"]).size().rename("tickets").reset_index()
    flows["topic_name"] = flows["topic"].map(topics.set_index("topic")["name"])
    cats = sorted(flows[Y].unique())
    tnames = list(topics.sort_values("tickets", ascending=False)["name"])
    idx = {n: i for i, n in enumerate(cats + tnames)}
    fig = go.Figure(go.Sankey(
        node=dict(label=[LABEL[c_] for c_ in cats] + tnames, pad=14, thickness=16,
                  color=[COLORS[c_] for c_ in cats] + ["#CBD5E1"] * len(tnames)),
        link=dict(source=flows[Y].map(idx), target=flows["topic_name"].map(idx), value=flows["tickets"],
                  color=[rgba(COLORS[c_], 0.35) for c_ in flows[Y]])))
    fig.update_layout(title="Categories (left) → cross-category topics (right), in tickets")
    fig_show(fig, 520)
    st.dataframe(topics[["name", "tickets", "categories", "products", "example"]], hide_index=True, **W,
                 column_config={"name": "Topic", "example": st.column_config.TextColumn("Example", width="large")})
    c = st.columns(2)
    with c[0]:
        insight("🔐 The <b>login topic</b> holds 3,460 tickets across 5 categories, although only 2,522 are labelled "
                "account access: login-related bugs, slowness, set-up questions and vulnerabilities belong here too.")
    with c[1]:
        insight("💳 The <b>billing topic</b> holds 2,992 tickets vs 2,476 labelled billing problem. Reporting by category "
                "alone hides these subject areas.", "#F59E0B")

    with st.expander("How BGE and TF-IDF + LSA see similarity, and where the themes differ"):
        sim = F["clustering_similarity"].rename(columns={"Unnamed: 0": "representation"})
        long = sim.melt("representation", var_name="pair", value_name="similarity")
        fig = px.bar(long, x="pair", y="similarity", color="representation", barmode="group", text="similarity",
                     color_discrete_sequence=["#A1A1AA", "#4F46E5"], title="Average cosine similarity of sentence pairs")
        fig.update_traces(texttemplate="%{y:.2f}")
        fig.update_layout(xaxis_title=None, yaxis_title=None)
        fig_show(fig, 340, legend_bottom=True)
        st.dataframe(agree, hide_index=True, **W)


# ------------------------------------------------------------------ anomalies
def page_anomalies():
    v = sidebar_filters()
    hero("🚨 Anomalies and hotspots", "Unusual ticket volumes, groups that do worse than average, and unusual individual tickets.",
         ["robust z-score", "proportion tests (Bonferroni)", "Isolation Forest"])
    va = F["volume_anomalies"].copy()
    va["period"] = pd.to_datetime(va["period"])
    c = st.columns(4)
    kpi(c[0], "Flagged periods", f"{int(va['is_anomaly'].sum())}", "robust z-score above 3", "#EF4444")
    hs = F["hotspots"]
    kpi(c[1], "Significant hotspots", f"{int(hs['significant'].sum())}", "Bonferroni-corrected", "#F59E0B")
    ta = F["ticket_anomalies"]
    kpi(c[2], "Unusual tickets", f"{len(ta):,}", "Isolation Forest on resolved tickets", "#8B5CF6")
    kpi(c[3], "Detector check", F["anomaly_validation"].iloc[-1]["result"], F["anomaly_validation"].iloc[-1]["check"], "#10B981")

    series = st.selectbox("Series", ["ALL (weekly)"] + [f"{c_} (monthly)" for c_ in CLASSES])
    s = va[va["category"] == series.split(" ")[0]]
    fig = go.Figure([
        go.Scatter(x=s["period"], y=s["expected"], name="expected", line=dict(dash="dash", color="#9CA3AF")),
        go.Scatter(x=s["period"], y=s["count"], name="actual", line=dict(color="#4F46E5", width=2.5), fill="tonexty",
                   fillcolor="rgba(79,70,229,0.08)"),
        go.Scatter(x=s.loc[s["is_anomaly"], "period"], y=s.loc[s["is_anomaly"], "count"], mode="markers", name="anomaly",
                   marker=dict(color="#EF4444", size=13, line=dict(color="#fff", width=2)))])
    fig.update_layout(title=f"Ticket volume · {series}", yaxis_rangemode="tozero")
    fig_show(fig, 380, legend_bottom=True)

    left, right = st.columns([1.3, 1])
    sig = hs[hs["significant"]].sort_values("z").copy()
    sig["segment"] = sig["metric"].str.replace("_", " ") + " · " + sig["value"].astype(str)
    fig = go.Figure()
    fig.add_bar(y=sig["segment"], x=sig["rate"], orientation="h", name="segment rate", marker_color="#EF4444",
                text=sig["rate"].map("{:.0%}".format), textposition="outside")
    fig.add_scatter(y=sig["segment"], x=sig["overall"], mode="markers", name="overall rate",
                    marker=dict(symbol="line-ns", size=22, line=dict(width=3, color="#111827")))
    fig.update_layout(title="Hotspots: segments significantly worse than overall", xaxis_tickformat=".0%")
    fig_show(fig, 420, legend_bottom=True, container=left)
    with right:
        pat = ta["why"].value_counts().head(6).rename_axis("pattern").reset_index(name="tickets")
        fig = px.bar(pat.iloc[::-1], x="tickets", y="pattern", orientation="h", text="tickets",
                     title="Most common patterns in unusual tickets")
        fig.update_traces(marker_color="#8B5CF6")
        fig.update_layout(xaxis_title=None, yaxis_title=None)
        fig_show(fig, 420)
    st.markdown("**Unusual individual tickets** (highest anomaly score first)")
    st.dataframe(ta[ta["ticket_id"].isin(v["ticket_id"])].sort_values("anomaly_score", ascending=False).round(2),
                 hide_index=True, **W, height=300)
    st.markdown("**Detector check** (planted spikes in a copy of the data)")
    st.dataframe(F["anomaly_validation"], hide_index=True, **W)


# ------------------------------------------------------------------ assistant
SUGGESTIONS = ["Which model is best and how much better is it than TF-IDF?",
               "Which categories have the most unhappy customers?",
               "How much does the BGE model reduce manual review work?",
               "Are there recurring issues that span several categories?",
               "Can we trust the SHAP explanations?",
               "Is there data leakage in this dataset?",
               "What drives resolution time?",
               "What are the limitations of this project?"]


def page_assistant():
    hero("💬 Ask the data",
         "A RAG assistant over every analysis in this project: it finds the relevant results and answers with sources.",
         [f"LLM: {'Mistral' if rag.llm_provider() == 'mistral' else rag.llm_provider()}",
          f"{len(F['knowledge_base'])} knowledge documents", "hybrid semantic search", "number check"])
    retriever = get_retriever()
    if "chat" not in st.session_state:
        st.session_state["chat"] = []
    chat = st.session_state["chat"]

    with st.sidebar:
        st.markdown("#### 💬 Assistant")
        provider = rag.llm_provider()
        if provider == "offline":
            st.warning("No API key found: answers quote the best documents directly.")
        else:
            st.success(f"Connected to {provider.capitalize()}" +
                       (f" · {config.MISTRAL_MODELS[0]}" if provider == "mistral" else ""))
        st.caption(f"Search: {retriever.mode}")
        if st.button("🧹 Clear conversation", **W):
            st.session_state["chat"] = []
            st.rerun()

    pick = None
    if not chat:
        st.markdown("**Try one of these questions**")
        pick = st.pills("Suggestions", SUGGESTIONS, label_visibility="collapsed", key="sugg")

    for turn in chat:
        render_turn(turn)

    question = st.chat_input("Ask about categories, models, SHAP, themes, anomalies, data quality...") or pick
    if question:
        history = [{"role": t["role"], "content": t["content"]} for t in chat]
        user = {"role": "user", "content": question}
        render_turn(user)
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Searching the analyses and writing an answer..."):
                res = rag.answer(question, retriever, k=5, history=history)
        chat += [user, {"role": "assistant", "content": res["answer"], "meta": res}]
        st.rerun()


def render_turn(turn):
    if turn["role"] == "user":
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(turn["content"])
        return
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(turn["content"])
        meta = turn.get("meta")
        if meta:
            badge = "✅ all numbers found in the sources" if meta["numbers_grounded"] else \
                "⚠️ some numbers were not found in the sources, please double-check"
            st.caption(f"{badge} · {meta['mode']}")
            with st.expander(f"📚 Sources ({len(meta['sources'])})"):
                kb = F["knowledge_base"].set_index("doc_id")
                for doc_id, score in zip(meta["sources"], meta["scores"]):
                    st.markdown(f"**[{doc_id}]** {kb.loc[doc_id, 'title']} · relevance {score:.2f}")
                    st.caption(kb.loc[doc_id, "text"][:600] + ("..." if len(kb.loc[doc_id, "text"]) > 600 else ""))


# ------------------------------------------------------------------ data and methodology
def page_data():
    hero("📚 Data & methodology", "What the data looks like, how the analysis was done, and what the assistant knows.",
         ["synthetic Kaggle dataset", "stratified 20k sample", "leakage handled"])
    prof, method, kb_tab, report = st.tabs(["📊 Data profile", "🛠️ Methodology", "🗂️ Knowledge base", "📄 Evaluation report"])
    with prof:
        cols = ["priority", "channel", "product_area", "customer_segment", "platform", "region", "sla_plan", "status"]
        grid = st.columns(4)
        for i, col in enumerate(cols):
            d = DATA[col].value_counts().rename_axis(col).reset_index(name="tickets")
            fig = px.bar(d, x="tickets", y=col, orientation="h", title=col.replace("_", " ").capitalize())
            fig.update_traces(marker_color=px.colors.qualitative.Pastel[i % 10], marker_line_width=0)
            fig.update_layout(xaxis_title=None, yaxis_title=None, yaxis={"categoryorder": "total ascending"})
            fig_show(fig, 260, container=grid[i % 4])
        insight("The ticket fields are evenly spread and carry <b>no information about the category</b>: within every "
                "product area each category has the same share. That's why the classifiers read only the description.")
    with method:
        steps = [("01", "Sampling, cleaning, EDA", "Stratified 20k sample, quality checks, leakage check"),
                 ("02 / 2b", "Classification", "TF-IDF and sentence embeddings (MiniLM, MPNet, BGE) × 5 classifiers"),
                 ("03 / 3b", "Explainability", "Exact SHAP for both models, faithfulness test"),
                 ("04 / 4b", "Recurring issues", "K-Means themes and cross-category topics"),
                 ("05", "Anomalies", "Volume z-scores, hotspot tests, Isolation Forest"),
                 ("06", "GenAI assistant", "Hybrid RAG over all results, Mistral, number check")]
        c = st.columns(len(steps))
        for col, (n, title, desc) in zip(c, steps):
            col.markdown(f'<div class="kpi"><div class="bar" style="background:#4F46E5"></div><div class="label">Notebook {n}'
                         f'</div><div style="font-weight:800;margin:4px 0">{title}</div><div class="sub">{desc}</div></div>',
                         unsafe_allow_html=True)
        st.markdown("")
        st.markdown("""
- **Data:** a stratified sample of 20,000 tickets from a synthetic dataset of 100,000 IT support tickets (seed 42), 8 balanced categories.
- **Leakage handled:** only 96 unique sentences exist, so all splits are **by message**. `agent_first_reply` and `resolution_summary` reveal the label and are never used.
- **Models:** the best model is BGE sentence embeddings + Logistic Regression (macro F1 0.82); the original TF-IDF + Logistic Regression scores 0.54 on the same test.
- **Confidence rule:** below 60% confidence a ticket goes to Manual Triage instead of a team.
- **Assumptions:** timestamps are synthetic; the category → team routing table is our own (no assignment-group column); `csat_score = 0` means not rated.
- **Out of scope:** the system suggests categories and explains results; it never closes or changes tickets.
""")
    with kb_tab:
        kb = F["knowledge_base"]
        c = st.columns([2, 1])
        q = c[0].text_input("Search the knowledge base", placeholder="e.g. leakage, SHAP, billing")
        src = c[1].multiselect("Source", sorted(kb["source"].unique()))
        d = kb
        if q:
            d = d[d["title"].str.contains(q, case=False) | d["text"].str.contains(q, case=False)]
        if src:
            d = d[d["source"].isin(src)]
        st.caption(f"{len(d)} of {len(kb)} documents")
        st.dataframe(d, hide_index=True, **W, height=420,
                     column_config={"text": st.column_config.TextColumn("Text", width="large")})
        st.markdown("**Retrieval test** (does the search find the right document?)")
        st.dataframe(F["rag_retrieval_eval"], hide_index=True, **W)
    with report:
        path = config.ROOT / "reports" / "evaluation_report.md"
        if path.exists():
            st.markdown(path.read_text())


# ================================================================== navigation
START = os.getenv("TICKET_APP_START_PAGE", "overview")      # optional: open the app on another page


def page(fn, title, icon, url):
    return st.Page(fn, title=title, icon=icon, url_path=url, default=(url == START))


pages = {
    "Dashboard": [page(page_overview, "Overview", ":material/space_dashboard:", "overview"),
                  page(page_themes, "Recurring issues", ":material/hub:", "themes"),
                  page(page_anomalies, "Anomalies", ":material/warning:", "anomalies")],
    "AI & models": [page(page_classify, "Classify a ticket", ":material/smart_toy:", "classify"),
                    page(page_assistant, "Ask the data", ":material/forum:", "assistant"),
                    page(page_models, "Model performance", ":material/insights:", "models"),
                    page(page_explain, "Explainability", ":material/psychology:", "explain")],
    "About": [page(page_data, "Data & methodology", ":material/description:", "data")],
}
st.logo(str(config.ROOT / "assets" / "logo.svg"), size="large", icon_image=str(config.ROOT / "assets" / "icon.svg"))
pg = st.navigation(pages)
pg.run()
