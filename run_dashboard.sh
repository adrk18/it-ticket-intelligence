#!/bin/bash
# Mac / Linux: install requirements (first time) and start the dashboard
cd "$(dirname "$0")"
pip install -r requirements.txt
streamlit run app.py
