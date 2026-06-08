# Secure Installer and Bootstrapper for Jarvis Local Assistant
# Runs on Windows PowerShell (Bypass execution policy)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Enforce UTF-8 globally to prevent mixed UTF-16LE / cp1252 corruption in logs and artifacts
$OutputEncoding = [Console]::OutputEncoding = [Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "       JARVIS LOCAL ASSISTANT SETUP          " -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# 1. Check Python Version
Write-Host "[1/6] Verifying Python installation..." -ForegroundColor Yellow
$PythonPath = $null
$PythonVersionString = $null

function Test-PythonVersion {
    param([string]$cmd)
    try {
        $version = & $cmd --version 2>&1
        if ($version -match "Python\s+(\d+)\.(\d+)") {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            if ($major -gt 3 -or ($major -eq 3 -and $minor -ge 11)) {
                return @{ Path = $cmd; Version = $version }
            } else {
                return @{ Path = $null; Version = $version; Error = "Python 3.11+ is required. Found Python $major.$minor" }
            }
        }
    } catch {}
    return $null
}

$res = Test-PythonVersion "python"
if ($null -eq $res -or $null -eq $res.Path) {
    $resPy = Test-PythonVersion "py"
    if ($null -ne $resPy -and $null -ne $resPy.Path) {
        $res = $resPy
    }
}

if ($null -eq $res) {
    throw "Python not found. Please install Python 3.11+ and add it to your PATH."
} elseif ($null -eq $res.Path) {
    throw $res.Error
}

$PythonPath = $res.Path
$PythonVersionString = $res.Version
Write-Host "Found Python: $PythonVersionString" -ForegroundColor Green


# 2. Virtual Environment Setup
Write-Host "[2/6] Setting up Virtual Environment..." -ForegroundColor Yellow
$VenvDir = Join-Path $PSScriptRoot "jarvis_env"
if (Test-Path -Path $VenvDir) {
    Write-Host "Virtual environment 'jarvis_env' already exists. Skipping creation." -ForegroundColor Green
} else {
    Write-Host "Creating virtual environment 'jarvis_env'..." -ForegroundColor Green
    & $PythonPath -m venv jarvis_env
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path -Path $VenvPython)) {
    throw "Failed to locate python interpreter inside virtual environment at $VenvPython"
}

# 3. Upgrade pip & Install dependencies
Write-Host "[3/6] Installing dependencies..." -ForegroundColor Yellow
& $VenvPython -m pip install --upgrade pip

Write-Host "Installing dependencies from requirements.lock (this may take a few minutes)..." -ForegroundColor Green
& $VenvPython -m pip install -r requirements.lock

# 4. Handle Configuration Environment File
Write-Host "[4/6] Initializing configuration (.env)..." -ForegroundColor Yellow
$EnvFile = Join-Path $PSScriptRoot ".env"
$EnvExample = Join-Path $PSScriptRoot ".env.example"

if (-not (Test-Path -Path $EnvFile)) {
    if (Test-Path -Path $EnvExample) {
        Copy-Item -Path $EnvExample -Destination $EnvFile
        Write-Host "Copied .env.example to .env" -ForegroundColor Green
    } else {
        New-Item -Path $EnvFile -ItemType File | Out-Null
        Write-Host "Created new empty .env file" -ForegroundColor Green
    }
}

# Generate secure random secret key and admin credentials if not set
$EnvContent = Get-Content -Path $EnvFile -Raw
if ($null -eq $EnvContent) { $EnvContent = "" }

# 1. JARVIS_SECRET_KEY
if ($EnvContent -notmatch "JARVIS_SECRET_KEY=") {
    $bytes = New-Object Byte[] 32
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $SecretKey = [Convert]::ToBase64String($bytes)
    Add-Content -Path $EnvFile -Value "`nJARVIS_SECRET_KEY=`"$SecretKey`""
    Write-Host "Generated random secure JARVIS_SECRET_KEY in .env" -ForegroundColor Green
}

# 2. JARVIS_ADMIN_USER and JARVIS_ADMIN_PASSWORD
$AdminUser = "admin"
$AdminPassword = $null
if ($EnvContent -notmatch "JARVIS_ADMIN_USER=") {
    # Generate secure random password
    $charList = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
    $random = New-Object System.Random
    $AdminPassword = -join (1..16 | ForEach-Object { $charList[$random.Next(0, $charList.Length)] })

    Add-Content -Path $EnvFile -Value "`nJARVIS_ADMIN_USER=`"$AdminUser`"`nJARVIS_ADMIN_PASSWORD=`"$AdminPassword`""
    Write-Host "Generated administrative access credentials in .env:" -ForegroundColor Green
    Write-Host "  Username: $AdminUser" -ForegroundColor Cyan
    Write-Host "  Password: [SECURELY SAVED IN .env]" -ForegroundColor Cyan
    Write-Host "Check your .env file for the password. Keep it safe!" -ForegroundColor Yellow
}

# 3. JARVIS_DASHBOARD_TOKEN (for API backward-compatibility)
if ($EnvContent -notmatch "JARVIS_DASHBOARD_TOKEN=") {
    $bytes = New-Object Byte[] 16
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $Token = [Convert]::ToBase64String($bytes).Replace('=', '').Replace('/', '').Replace('+', '')
    Add-Content -Path $EnvFile -Value "`nJARVIS_DASHBOARD_TOKEN=`"$Token`""
    Write-Host "Generated random secure JARVIS_DASHBOARD_TOKEN in .env" -ForegroundColor Green
}

# 5. Verify Ollama
Write-Host "[5/6] Checking Ollama Status..." -ForegroundColor Yellow
$OllamaProcess = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
if ($null -eq $OllamaProcess) {
    Write-Host "Ollama process is not running. Please download and start Ollama from https://ollama.com/" -ForegroundColor Yellow
} else {
    Write-Host "Ollama is running." -ForegroundColor Green
}

# 6. Run Diagnostic Check
Write-Host "[6/6] Running system diagnostics..." -ForegroundColor Yellow
& $VenvPython main.py --health-check
if ($LASTEXITCODE -ne 0) {
    Write-Host "Diagnostics returned non-zero ($LASTEXITCODE). Check your configuration." -ForegroundColor Red
}

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "Setup Completed Successfully!" -ForegroundColor Green
Write-Host "Start the dashboard using: " -ForegroundColor Yellow
Write-Host "  powershell -ExecutionPolicy Bypass .\Start.ps1 --gui" -ForegroundColor Green
if ($null -ne $AdminPassword) {
    Write-Host "Initial Dashboard Login Credentials:" -ForegroundColor Yellow
    Write-Host "  Username: $AdminUser" -ForegroundColor Cyan
    Write-Host "  Password: [SECURELY SAVED IN .env]" -ForegroundColor Cyan
}
Write-Host "=============================================" -ForegroundColor Cyan
