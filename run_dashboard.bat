@echo off
REM Windows: install requirements (first time) and start the dashboard
cd /d "%~dp0"
pip install -r requirements.txt
streamlit run app.py
pause
