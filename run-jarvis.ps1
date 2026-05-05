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

$python = Resolve-ProjectPython
$entrypoint = Join-Path $PSScriptRoot "main.py"

if (-not (Test-Path -LiteralPath $entrypoint)) {
    throw "Cannot find main.py at $entrypoint"
}

& $python $entrypoint @JarvisArgs
exit $LASTEXITCODE
