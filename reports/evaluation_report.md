# Evaluation Report: IT Support Ticket Classification

All numbers come from `data/processed/` and the notebooks in `notebooks/`. Charts are in `reports/figures/`.

## 1. Data and class distribution

- Stratified sample of **20,000** tickets from a synthetic dataset of 100,000 (seed 42); category shares match the full data.
- **8 categories**, balanced: largest how_to 12.7%, smallest bug 12.3% (imbalance ratio 1.03). No resampling needed.
- Only **96 unique messages**; each always carries the same label. account_access, billing_problem and other have just 3 sentences each.
- Cleaning: `csat_score = 0` → not rated; missing region → "Unknown"; resolution fields left empty for unfinished tickets.
- Figures: `01_target.png`, `06_class_balance.png`.

## 2. Split strategy and leakage

| Issue | Decision |
|---|---|
| Same 96 sentences repeat across tickets | **Split by message**: test sentences are never seen in training. A random split gives a misleading macro F1 of 1.00. |
| `agent_first_reply`, `resolution_summary` map one-to-one to the label | Excluded from model inputs |
| Status, sentiment, CSAT, resolution time are known only after handling | Excluded from model inputs; used in analytics only |

Hold-out test: 20% of messages per category (4,523 tickets from 20 messages). Stability check: grouped 5-fold cross-validation.

## 3. Baseline and model comparison

Test set (unseen messages):

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | Macro ROC-AUC |
|---|---|---|---|---|---|---|
| Majority baseline | 0.049 | 0.006 | 0.125 | 0.012 | 0.005 | 0.500 |
| Naive Bayes | 0.630 | 0.573 | 0.750 | 0.611 | 0.548 | 0.915 |
| Linear SVM | 0.630 | 0.573 | 0.750 | 0.611 | 0.548 | 0.886 |
| Logistic Regression | 0.630 | 0.573 | 0.750 | 0.611 | 0.548 | 0.886 |

Grouped 5-fold CV macro F1: 0.591 ± 0.051 for all three real models (baseline 0.002).
**Chosen: Logistic Regression.** It ties on F1, gives probabilities for the confidence score, and its SHAP values are exact.

## 4. Final model: per-class results (out-of-fold, all 20,000 tickets)

| Category | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| account_access | 0.00 | 0.00 | 0.00 | 0.65 |
| billing_problem | 0.50 | 0.34 | 0.41 | 0.89 |
| bug | 0.75 | 1.00 | 0.86 | 1.00 |
| feature_request | 1.00 | 1.00 | 1.00 | 1.00 |
| how_to | 0.51 | 1.00 | 0.67 | 1.00 |
| other | 0.00 | 0.00 | 0.00 | 0.64 |
| performance | 1.00 | 1.00 | 1.00 | 1.00 |
| security_concern | 0.49 | 0.33 | 0.40 | 0.72 |
| **Macro average** | **0.53** | **0.58** | **0.54** | **0.86** |
| **Weighted average** | 0.53 | 0.58 | 0.54 | |

Accuracy 0.585. **Minority categories:** the three 3-sentence classes are the weakest (account_access and other F1 0.00, billing_problem 0.41).

## 5. Confusion and error analysis

Most frequent confusions (out-of-fold):

| Actual → Predicted | Tickets | Why |
|---|---|---|
| other → how_to | 1,659 | "I need help but I'm not sure..." looks like "I need help setting up..." |
| account_access → security_concern | 863 | "account was locked after failed **login** attempts" vs "suspicious **login** on my account" |
| security_concern → other | 832 | "We need details about your data encryption" shares "need", "details" with vague requests |
| account_access → billing_problem / bug | 830 / 829 | Unseen password and 2FA wording shares few words with the other access sentences |
| billing_problem → how_to / account_access | 821 / 814 | Unseen billing wording ("downgraded my plan", "invoice") |

The model matches words rather than meaning. With so few sentences per class, unseen wording in small classes is easily confused. Figure: `22_confusion_matrix.png`.

## 6. Low-confidence handling

- Rule: confidence below **60%** → Manual Triage queue.
- 45.7% of tickets fall below the threshold (out-of-fold).
- **Auto-routed tickets are 92% correct; reviewed tickets only 18%**, so the rule catches most mistakes.
- Confidence is only partly calibrated (e.g. some correct billing predictions sit below 40%), so the threshold should be tuned on real data. The dashboard has a slider for this.
- Figure: `24_confidence.png`.

## 7. Worked examples (model trained on training messages only)

| Case | Message | Prediction | Confidence | Decision |
|---|---|---|---|---|
| Common (seen) | "I need help setting up data export for my team." | how_to | 0.99 | Auto-route to Service Desk (L1) |
| Ambiguous | "The billing page is very slow and keeps showing an error." | performance (runner-up bug) | 0.71 | Auto-route to Platform / SRE |
| Minority class, new wording | "My account was locked after multiple failed login attempts." | security_concern | 0.47 | Manual review |
| Out of scope | "The coffee machine on the third floor is broken." | bug | 0.46 | Manual review |

## 8. Explainability

Exact SHAP values for the linear model (verified to add up to the model's scores). Key words per class are sensible
(e.g. "charged twice", "subscription" → billing; "timing out", "queries" → performance; "suspicious", "vulnerability" → security).
Filler words ("my", "the", "after") also carry weight because of template sentences. Figures: `30–32_shap_*.png`.

## 9. Recurring issues and anomalies (summary)

- **24 recurring themes** from text embeddings + K-Means within categories, e.g. pages crashing, changes not saving, queries timing out. Themes are coherent (cohesion 0.76–1.00).
- **Volume:** no weekly spikes; 1 category-month flag (in line with chance). The detector catches every planted spike of +60% or more.
- **Hotspots:** performance, security_concern, billing_problem and account_access are about 60% negative vs 42% overall.
- **Unusual tickets:** 102 resolved tickets flagged by Isolation Forest, mostly reopened, often high priority.

## 10. Limitations

- Synthetic data: generated labels and timestamps, even distributions, hard resolution-time limits per priority.
- Only 96 unique sentences, so the scores measure generalisation to new wording on very little variety. Real data should score higher.
- The routing table (category → queue) is assumed; the data has no assignment group.
- TF-IDF matches words, not meaning; sentence embeddings or more varied training text are the main improvements.
- The GenAI assistant uses keyword retrieval; LLM answers need an API key (offline mode otherwise).
