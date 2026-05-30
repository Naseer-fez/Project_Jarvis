[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PytestArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$cppRunner = Join-Path $PSScriptRoot "bin\test_runner.exe"
$usePytestOnly = $PytestArgs -contains "--pytest"

if ((Test-Path -LiteralPath $cppRunner) -and -not $usePytestOnly) {
    $forwardArgs = $PytestArgs | Where-Object { $_ -ne "--pytest" }
    & $cppRunner $forwardArgs
    exit $LASTEXITCODE
}

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

& $python -c "import pytest" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "pytest is not installed for $python. Install requirements/dev.txt or use the project's virtual environment."
}

$forwardArgs = $PytestArgs | Where-Object { $_ -ne "--pytest" }
& $python -m pytest $forwardArgs
exit $LASTEXITCODE
