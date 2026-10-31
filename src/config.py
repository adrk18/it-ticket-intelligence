"""Shared settings used by the notebooks and the dashboard. Change things here, not in the code."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = ROOT / "data" / "raw" / "synthetic_it_support_tickets.csv.xlsx"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIGURES = ROOT / "reports" / "figures"
for folder in (PROCESSED, MODELS, FIGURES):
    folder.mkdir(parents=True, exist_ok=True)

SEED = 42
SAMPLE_SIZE = 20_000

TARGET = "issue_type"
TEXT = "initial_message"

# A ticket goes to manual review when the model is less sure than this
CONFIDENCE_THRESHOLD = 0.60

# The dataset has no assignment-group column, so this routing table is our own assumption
ROUTING = {
    "account_access": "Identity & Access Team",
    "billing_problem": "Billing Operations",
    "bug": "Application Support (L2)",
    "feature_request": "Product Management",
    "how_to": "Service Desk (L1)",
    "other": "Manual Triage",
    "performance": "Platform / SRE",
    "security_concern": "Security Operations",
}
REVIEW_QUEUE = "Manual Triage"

# Product names that appear inside the ticket text
PRODUCTS = ["analytics dashboard", "api integration", "billing", "data export",
            "login auth", "mobile app", "notifications"]

# GenAI settings (keys are read from environment variables or the .env file, never stored here)
ANTHROPIC_MODEL = "claude-haiku-5-5"
OPENAI_MODEL = "gpt-4o-mini"
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
# tried in order; the next one is used if a model is rate-limited or not in the subscription tier
MISTRAL_MODELS = ["ministral-14b-latest", "mistral-small-latest", "ministral-8b-latest"]

# Sentence-embedding model used by the best classifier (notebook 02b) and the assistant's semantic search
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
ENV_FILE = ROOT / ".env"
