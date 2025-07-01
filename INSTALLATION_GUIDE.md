# 🚀 Mimir Installation Guide

## Quick Start (Windows)

### Option 0: Install Dependencies First (NEW!)
```cmd
python install_dependencies.py
```
This installs all essential packages needed for both the scraper and web interface.

### Option 1: Using Batch File (Easiest)
```cmd
start_netflix_ui.bat
```

### Option 2: Using PowerShell
```powershell
# If you get execution policy error, run this first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then run:
.\start_netflix_ui.ps1
```

### Option 3: Manual Installation

1. **Install Required Packages**
   ```cmd
   python -m pip install --user flask flask-cors
   ```
   
   If that doesn't work, try:
   - Run Command Prompt as Administrator
   - Or use: `python -m pip install flask flask-cors`

2. **Create Database** (if not exists)
   ```cmd
   python db_manager.py
   ```

3. **Start the Interface**
   ```cmd
   python web_interface.py
   ```

4. **Open Browser**
   Visit: http://localhost:5000

## Troubleshooting

### "Permission Denied" Error
- Use `--user` flag: `python -m pip install --user flask flask-cors`
- Or run Command Prompt as Administrator
- Or use a virtual environment (see below)

### Virtual Environment (Recommended for Development)
```cmd
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# Install packages
pip install flask flask-cors

# Run the app
python web_interface.py
```

### "Module Not Found" Error
Make sure you're in the Mimir directory:
```cmd
cd C:\Projects\Mimir
```

### Port Already in Use
Change the port in `web_interface.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change 5000 to 5001
```

## First Time Setup

1. **Run the Scraper First**
   ```cmd
   python scraper.py --run
   ```
   This populates the database with articles.

2. **Configure Your Keywords**
   Edit `config.json` to set your interests:
   ```json
   {
     "keywords": ["AI", "security", "Apple", "Microsoft"]
   }
   ```

3. **Add News Sources**
   - Open the interface
   - Click the ⚙️ icon
   - Add RSS feeds or websites

## System Requirements

- Python 3.7 or higher
- Windows/Linux/macOS
- Modern web browser (Chrome, Firefox, Edge)
- 100MB free disk space

## Need Help?

If you're still having issues:
1. Check if Python is installed: `python --version`
2. Update pip: `python -m pip install --upgrade pip`
3. Check the error messages carefully
4. Make sure you're in the correct directory

Happy news browsing! 🎬📰