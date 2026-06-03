Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

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
        return $systemPython.Source
    }

    throw "No Python interpreter found. Create a virtual environment or install Python first."
}

$python = Resolve-ProjectPython
$launcher = Join-Path $PSScriptRoot "run_tests.py"
& $python $launcher @PytestArgs
exit $LASTEXITCODE
