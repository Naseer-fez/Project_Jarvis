[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$JarvisArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-ProjectPython {
    $candidates = @(
        (Join-Path $PSScriptRoot "jarvis_env\Scripts\python.exe"),
        (Join-Path $PSScriptRoot ".venv\Scripts\python.exe"),
        (Join-Path $PSScriptRoot "venv\Scripts\python.exe")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }

    $systemPython = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $systemPython) {
        return $systemPython.Source
    }

    throw "No Python interpreter found. Create a virtual environment or install Python first."
}

function Ensure-OllamaRunning {
    Write-Host "Checking if Ollama is running..." -ForegroundColor Cyan
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:11434/" -Method Get -ErrorAction Stop
        if ($response -match "Ollama is running") {
            Write-Host "Ollama is up and running." -ForegroundColor Green
            return
        }
    } catch {
        Write-Host "Ollama is not running on localhost:11434." -ForegroundColor Yellow
        $ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
        if ($null -eq $ollamaCmd) {
            Write-Host "WARNING: 'ollama' command not found. Local models will fail, and Jarvis will rely on cloud fallbacks." -ForegroundColor Red
            Write-Host "Download Ollama from https://ollama.com/download if you want local execution." -ForegroundColor Yellow
            return
        }
        
        Write-Host "Attempting to start Ollama in the background..." -ForegroundColor Cyan
        Start-Process -WindowStyle Hidden -FilePath "ollama" -ArgumentList "serve"
        
        # Wait a few seconds for it to start
        Start-Sleep -Seconds 3
        try {
            $response = Invoke-RestMethod -Uri "http://localhost:11434/" -Method Get -ErrorAction Stop
            Write-Host "Ollama successfully started." -ForegroundColor Green
        } catch {
            Write-Host "WARNING: Failed to verify Ollama started successfully. Local models might fail." -ForegroundColor Red
        }
    }
}

Ensure-OllamaRunning

$python = Resolve-ProjectPython
$entrypoint = Join-Path $PSScriptRoot "main.py"

if (-not (Test-Path -LiteralPath $entrypoint)) {
    throw "Cannot find main.py at $entrypoint"
}

& $python $entrypoint @JarvisArgs
exit $LASTEXITCODE
