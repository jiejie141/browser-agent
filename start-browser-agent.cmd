@echo off
rem Start browser-agent web console (persistent, survives IDE turn end).
rem Keep this file pure ASCII: non-ASCII in .cmd gets garbled by codepage.
rem NOTE: must use the project .venv (playwright lives there, not in global env).
cd /d C:\Users\pc\Desktop\browser-agent
set PYTHONPATH=src
C:\Users\pc\Desktop\browser-agent\.venv\Scripts\python.exe -m uvicorn bagent.api:app --host 127.0.0.1 --port 8000
