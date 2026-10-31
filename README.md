# IT Support Ticket Classification and Analytics

HCLTech Future Ready Data Science Hackathon · Use case 4

Support teams receive thousands of tickets and sort them by hand, which slows down routing and hides problems that keep coming back.
This project reads a ticket's description, predicts its category with a confidence score, sends uncertain tickets to a person,
finds recurring issues and anomalies, and puts it all in a dashboard with a GenAI assistant you can ask questions in plain English.

## What's inside

| Step | Notebook | What it does | Main output |
|---|---|---|---|
| 1 | `01_sampling_cleaning_eda` | Takes a stratified 20,000-ticket sample, checks quality, cleans, and explores: univariate, bivariate, class imbalance, correlation, data leakage | `tickets_clean.csv` |
| 2 | `02_classification_models` | TF-IDF features; compares Logistic Regression, Linear SVM, Naive Bayes and a baseline; precision, recall, F1, confusion matrix, ROC-AUC, confidence threshold | `predictions.csv`, `metrics.json`, `models/tfidf_logreg.joblib` |
| 2b | `02b_semantic_embeddings` | Replaces TF-IDF with sentence embeddings (MiniLM, MPNet, BGE) and compares LR, SVM, kNN, Random Forest on each; same splits as 02 | `embedding_cv_scores.csv`, `models/bge_logreg.joblib` |
| 3 | `03_shap_explainability` | Exact SHAP values for the linear model: global words per category and per-ticket explanations | `shap_global.csv`, `shap_local.csv` |
| 3b | `03b_shap_embeddings` | Exact word-level SHAP for the BGE + LR model (each word deleted and the sentence re-embedded), compared with notebook 03, plus a deletion test | `shap_bge_global.csv`, `shap_bge_local.csv` |
| 4 | `04_recurring_issues_clustering` | Text embeddings (TF-IDF + LSA) and K-Means to group similar descriptions into recurring themes | `clusters.csv`, `cluster_summary.csv` |
| 4b | `04b_clustering_embeddings` | Recurring themes and cross-category topics with BGE embeddings, compared with notebook 04 | `clusters_bge.csv`, `cluster_summary_bge.csv`, `topic_clusters_bge.csv` |
| 5 | `05_anomaly_detection` | Volume spikes (robust z-score), hotspots (proportion tests), unusual tickets (Isolation Forest), plus a detector test | `volume_anomalies.csv`, `hotspots.csv`, `ticket_anomalies.csv` |
| 6 | `06_genai_rag_assistant` | RAG assistant over the project's results, with an offline mode and a number check | `knowledge_base.csv` |
| – | `app.py` | Streamlit dashboard bringing everything together | – |

## How to run

```bash
pip install -r requirements.txt

# 1. put the dataset here (already included in the zip):
#    data/raw/synthetic_it_support_tickets.csv.xlsx

# 2. run the notebooks in order (01 → 06) from the notebooks/ folder, in Jupyter or with:
jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb

# 3. start the dashboard from the project folder
streamlit run app.py
```

The zip already contains all outputs, so you can start the dashboard straight away (step 3).
If the dashboard can't load the model (a scikit-learn version difference), just re-run notebook 02 once.

**Optional: real LLM answers.** Without a key, the assistant works offline. To use an LLM, set one of these before starting Jupyter or Streamlit:

```bash
export ANTHROPIC_API_KEY=...        # Claude
# or
export OPENAI_API_KEY=...           # OpenAI, or any compatible service:
export OPENAI_BASE_URL=...          # e.g. Groq or a local Ollama server
```
On Windows PowerShell use `$env:ANTHROPIC_API_KEY="..."`. Model names are in `src/config.py`.

## How it fits together

```
data/raw (Excel, 100k tickets)
   │  01: stratified 20k sample → quality checks → cleaning → EDA
   ▼
data/processed/tickets_clean.csv  ───────────────────────────────┐
   │                                                             │
   ├─ 02: TF-IDF + LR / SVM / NB → predictions, metrics, model   │
   ├─ 03: SHAP on the LR model    → word explanations            │  all results saved
   ├─ 04: embeddings + K-Means    → recurring themes             │  as CSV / JSON
   ├─ 05: z-scores, tests, IForest→ anomalies                    │
   └─ 06: knowledge base from all of the above → RAG assistant   │
                                                                 ▼
                                    app.py (Streamlit) reads the saved files only
```

Each notebook reads files and writes files; nothing depends on another notebook's code. Shared settings
(paths, seed, confidence threshold, routing table, model names) live in `src/config.py`. The assistant's code is in
`src/rag.py`, so the notebook and the dashboard use exactly the same logic.

## Key results

| Area | Result |
|---|---|
| Data | 20,000 tickets, 8 balanced categories (ratio 1.03), only 96 unique messages |
| Leakage | Random split gives a misleading macro F1 of 1.00. We split by message instead. `agent_first_reply` and `resolution_summary` give away the label and are excluded. |
| Model comparison | LR, SVM and Naive Bayes tie on macro F1 (0.59 in grouped CV); LR chosen for probabilities and exact SHAP |
| Final model (out-of-fold) | Macro F1 0.54, weighted F1 0.54, macro ROC-AUC 0.86 |
| Semantic embeddings (02b) | BGE + LR / SVM reach macro F1 0.81–0.82 in grouped CV vs 0.59 for TF-IDF; account_access 0.00 → 0.80, manual review 46% → 17% |
| Strong / weak classes | bug, feature_request, how_to, performance near perfect on new wording; account_access and other fail (3 sentences each) |
| Low confidence | Below 60% → manual review (46% of tickets). Auto-routed tickets are 92% correct; reviewed ones only 18%, so the rule catches most mistakes |
| SHAP | Sensible key words per class, plus filler words caused by template sentences (a stated limitation) |
| SHAP for embeddings (03b) | Key words in every category's top 3 (password, error, suspicious…); deleting the top SHAP word hurts the prediction 5× more than a random word; stop-word share similar to TF-IDF (26% vs 27%) |
| Recurring issues | 24 coherent themes (e.g. pages crashing, changes not saving, queries timing out) |
| Clustering with embeddings (04b) | 5 of 8 categories get the same themes as notebook 04; BGE merges login/password failures and security incidents; a new topic view shows a login area of 3,460 tickets across 5 categories |
| Anomalies | No real volume spikes (detector catches every planted spike of +60% or more); problem categories ~60% negative vs 42% overall |
| GenAI | Right source document in the top 4 for all 13 test questions; invented numbers are flagged |

**Why the scores look modest:** the test only contains wording the model has never seen, and the dataset has very few
different sentences per category. This is the honest number. The commonly reported ~100% comes from the same sentences appearing in training and test.

## Dashboard tabs

| Tab | Shows |
|---|---|
| Overview | KPIs, tickets per category and queue, monthly trend, resolution trend by priority, negative share by category |
| Classify a ticket | TF-IDF prediction, confidence, queue or manual review, top-5 probabilities, SHAP word chart, GenAI explanation |
| Model performance | F1, precision, recall, ROC-AUC, accuracy; model comparison and CV; per-class table and confusion matrix (live for the filter); confidence chart and threshold slider |
| Explainability | SHAP top words per category; explanation of any description |
| Recurring issues | Largest themes, representative descriptions, theme trend |
| Anomalies | Volume with flagged periods, detector test, hotspots, unusual tickets |
| GenAI assistant | Ask questions; answers cite their source documents; number check |
| Data & assumptions | Data notes, assumptions, retrieval test, counts check |

All counts come from one table with one row per ticket, and the sidebar checks that category counts add up to the filtered total.

## Assumptions and limitations

- The data is a synthetic Kaggle dataset. Labels come from its `issue_type` column and timestamps are generated.
- There is no assignment-group column, so the category → support queue table in `src/config.py` is our own assumption.
- `csat_score = 0` is treated as "not rated".
- Only 96 different messages exist, so the classifier is tested on a hard task: new wording. Real, more varied data would likely score higher.
- Notebooks 02b, 03b and 04b need `pip install sentence-transformers` and download three models (~0.6 GB) the first time. The dashboard still uses the TF-IDF model, SHAP and themes from notebooks 02–04, so it doesn't need PyTorch.
- Text embeddings in notebook 04 use TF-IDF + LSA so everything runs offline. Notebook 04 has a switch for `sentence-transformers` if it's installed.
- The `shap` library is optional: SHAP values for the linear model are computed exactly with the same formula `shap.LinearExplainer` uses, and notebook 03 cross-checks against the library if it's installed.
- The system suggests categories and routing and explains results; it never closes or changes tickets.

## Folder structure

```
it_ticket_project/
├── README.md                          start here
├── requirements.txt
├── app.py                             Streamlit dashboard
├── run_dashboard.bat / .sh            one click: install + start the dashboard (Windows / Mac-Linux)
├── run_notebooks.sh                   re-run all notebooks in order
│
├── docs/
│   └── problem_statement.pdf          the hackathon brief
│
├── notebooks/                         run in this order (already run, with outputs)
│   ├── 01_sampling_cleaning_eda.ipynb
│   ├── 02_classification_models.ipynb
│   ├── 02b_semantic_embeddings.ipynb   (needs sentence-transformers)
│   ├── 03_shap_explainability.ipynb
│   ├── 03b_shap_embeddings.ipynb       (needs sentence-transformers, ~5 min)
│   ├── 04_recurring_issues_clustering.ipynb
│   ├── 04b_clustering_embeddings.ipynb (needs sentence-transformers)
│   ├── 05_anomaly_detection.ipynb
│   └── 06_genai_rag_assistant.ipynb
│
├── src/                               code shared by the notebooks and the dashboard
│   ├── config.py                      paths, seed, threshold, routing table, model names
│   └── rag.py                         knowledge base, retrieval, LLM / offline answers, number check
│
├── data/
│   ├── raw/                           original Excel file (100,000 tickets)
│   └── processed/                     every result the dashboard reads
│       ├── tickets_clean.csv          (01) cleaned 20,000-ticket sample
│       ├── predictions.csv            (02) out-of-fold prediction + confidence for every ticket
│       ├── metrics.json, classification_report.csv, confusion_matrix.csv,
│       │   model_comparison.csv, cv_scores.csv                      (02)
│       ├── embedding_holdout_comparison.csv, embedding_cv_scores.csv,
│       │   embedding_per_class_f1.csv                               (02b)
│       ├── shap_global.csv, shap_local.csv                          (03)
│       ├── shap_bge_global.csv, shap_bge_local.csv                  (03b)
│       ├── clusters.csv, cluster_summary.csv                        (04)
│       ├── clusters_bge.csv, cluster_summary_bge.csv, topic_clusters_bge.csv (04b)
│       ├── volume_anomalies.csv, hotspots.csv, ticket_anomalies.csv,
│       │   anomaly_validation.csv                                   (05)
│       └── knowledge_base.csv, rag_retrieval_eval.csv               (06)
│
├── models/
│   ├── tfidf_logreg.joblib            final classifier (used by the dashboard)
│   ├── bge_logreg.joblib              BGE + Logistic Regression from 02b
│   └── shap_background.npy            average TF-IDF vector for SHAP
│
└── reports/
    ├── evaluation_report.md           class distribution, split, baselines, metrics, errors, examples, limitations
    └── figures/                       all 36 charts (01-13 EDA, 20-27 model, 30-35 SHAP, 40-46 themes, 50-51 anomalies)
```

**Brief deliverables → where to find them**

| Deliverable | Location |
|---|---|
| Classification prototype | `notebooks/02_…`, live in the dashboard's "Classify a ticket" tab |
| Analytics dashboard | `app.py` |
| Evaluation report | `reports/evaluation_report.md` |
| Worked examples | `reports/evaluation_report.md` section 7, `notebooks/02_…` section 5.5 |
| Technical documentation | this README |

## Reproducibility

Seed 42 everywhere (sampling, splits, models, clustering, anomaly tests). Re-running the notebooks in order recreates every file exactly.
