@echo off
cd /d "%~dp0"

echo Activating virtual environment...
call venv\Scripts\activate

echo Starting Flask server...
start cmd /k python app.py

echo Starting local web server for frontend (port 8000)...
start cmd /k python -m http.server 8000

timeout /t 2 >nul

echo Opening index.html in browser...
start http://localhost:8000/index.html

exit
