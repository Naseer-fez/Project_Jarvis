Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Enforce UTF-8 globally to prevent mixed UTF-16LE / cp1252 corruption in logs and artifacts
$OutputEncoding = [Console]::OutputEncoding = [Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

$PytestArgs = @($args)


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
        Write-Host "WARNING: No virtual environment found. Falling back to system Python." -ForegroundColor Yellow
        Write-Host "If you experience missing modules, please run .\install.ps1 first." -ForegroundColor Yellow
        return $systemPython.Source
    }

    throw "No Python interpreter found. Please run .\install.ps1 to set up the project."
}

$python = Resolve-ProjectPython
$launcher = Join-Path $PSScriptRoot "run_tests.py"
& $python $launcher @PytestArgs
exit $LASTEXITCODE
