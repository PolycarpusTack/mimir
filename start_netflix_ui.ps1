# Mimir Netflix-Style Interface Launcher (PowerShell)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " 🎬 Mimir Netflix-Style Interface Launcher" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Function to check if a module is installed
function Test-PythonModule {
    param($ModuleName)
    $result = & python -c "import $ModuleName" 2>&1
    return $?
}

# Check Flask installation
Write-Host "Checking requirements..." -ForegroundColor Green
$modulesToCheck = @("flask", "flask_cors")
$missingModules = @()

foreach ($module in $modulesToCheck) {
    if (-not (Test-PythonModule $module)) {
        $missingModules += $module.Replace("_", "-")
    }
}

if ($missingModules.Count -gt 0) {
    Write-Host "Missing packages: $($missingModules -join ', ')" -ForegroundColor Yellow
    Write-Host "Installing required packages..." -ForegroundColor Green
    
    try {
        & python -m pip install --user $missingModules
        Write-Host "Packages installed successfully!" -ForegroundColor Green
    }
    catch {
        Write-Host "`n⚠️  Could not install packages automatically." -ForegroundColor Red
        Write-Host "`nPlease install manually with:" -ForegroundColor Yellow
        Write-Host "  python -m pip install --user flask flask-cors" -ForegroundColor Cyan
        Write-Host "`nPress any key to exit..."
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        exit 1
    }
}

# Check database
Write-Host "`nChecking database..." -ForegroundColor Green
if (-not (Test-Path "news_scraper.db")) {
    Write-Host "Database not found. Creating new database..." -ForegroundColor Yellow
    & python db_manager.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error creating database!" -ForegroundColor Red
        Write-Host "Press any key to exit..."
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        exit 1
    }
    Write-Host "Database created successfully!" -ForegroundColor Green
} else {
    Write-Host "Database found." -ForegroundColor Green
}

# Start the server
Write-Host "`nStarting Netflix-style web interface..." -ForegroundColor Green
Write-Host "------------------------------------------------------------" -ForegroundColor Gray
Write-Host "Access the interface at: " -NoNewline
Write-Host "http://localhost:5000" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host "------------------------------------------------------------" -ForegroundColor Gray
Write-Host ""

# Open browser after 2 seconds
Start-Job -ScriptBlock {
    Start-Sleep -Seconds 2
    Start-Process "http://localhost:5000"
} | Out-Null

# Run Flask app
& python web_interface.py