@echo off
echo ============================================================
echo  Mimir Netflix-Style Interface Launcher (Windows)
echo ============================================================
echo.

REM Check if Flask is installed
python -c "import flask" 2>nul
if errorlevel 1 (
    echo Flask not found. Installing required packages...
    echo.
    python -m pip install --user flask flask-cors
    if errorlevel 1 (
        echo.
        echo ERROR: Could not install packages automatically.
        echo Please run as Administrator or use:
        echo    python -m pip install --user flask flask-cors
        echo.
        pause
        exit /b 1
    )
)

REM Check if database exists
if not exist news_scraper.db (
    echo Database not found. Creating new database...
    python db_manager.py
    if errorlevel 1 (
        echo Error creating database!
        pause
        exit /b 1
    )
)

echo.
echo Starting Netflix-style web interface...
echo ------------------------------------------------------------
echo Access the interface at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo ------------------------------------------------------------
echo.

REM Open browser after 2 seconds
start /b cmd /c "timeout /t 2 >nul && start http://localhost:5000"

REM Run the Flask app
python web_interface.py

pause