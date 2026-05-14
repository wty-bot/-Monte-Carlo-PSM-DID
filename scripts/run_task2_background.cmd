@echo off
cd /d "%~dp0\.."
call ".venv\Scripts\python.exe" "scripts\run_task2_estimate.py" --n-sims 500 1>> "outputs\logs\task2_run_stdout.log" 2>> "outputs\logs\task2_run_stderr.log"
