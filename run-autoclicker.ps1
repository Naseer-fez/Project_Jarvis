[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$Target,

    [Parameter(Mandatory=$false)]
    [double]$Interval = 5.0,

    [switch]$Continuous
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Enforce UTF-8 globally to prevent mixed UTF-16LE / cp1252 corruption in logs and artifacts
$OutputEncoding = [Console]::OutputEncoding = [Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

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

$python = Resolve-ProjectPython
$entrypoint = Join-Path $PSScriptRoot "core\tools\auto_clicker.py"

if (-not (Test-Path -LiteralPath $entrypoint)) {
    throw "Cannot find auto_clicker.py at $entrypoint"
}

$argsList = @("--target", $Target, "--interval", $Interval)
if ($Continuous) {
    $argsList += "--continuous"
}

& $python $entrypoint @argsList
exit $LASTEXITCODE
