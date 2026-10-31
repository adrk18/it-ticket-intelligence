#!/bin/bash
# Re-run all six notebooks in order (recreates everything in data/processed, models and reports/figures)
cd "$(dirname "$0")"
for nb in notebooks/0*.ipynb; do
  echo "Running $nb"
  jupyter nbconvert --to notebook --execute --inplace "$nb" || exit 1
done
echo "Done. Start the dashboard with: streamlit run app.py"
