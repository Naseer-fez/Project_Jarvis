#!/usr/bin/env python3
"""
Jarvis Master Test Launcher
Wraps the high-performance C++ multi-threaded test runner.
Handles automatic compilation of the C++ runner if source files change,
and falls back to standard pytest if a C++ compiler is not available.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
CPP_SRC = PROJECT_ROOT / "bin" / "test_runner.cpp"
BINARY_NAME = "test_runner.exe" if sys.platform == "win32" else "test_runner"
CPP_BIN = PROJECT_ROOT / "bin" / BINARY_NAME


def has_compiler() -> bool:
    """Check if g++ is installed on the system."""
    return shutil.which("g++") is not None


def compile_runner() -> bool:
    """Compile the C++ test runner."""
    print("\x1b[36mCompiling C++ test runner from source...\x1b[0m")
    
    # Platform specific flags
    if sys.platform == "win32":
        cmd = [
            "g++", "-std=c++17", "-O3",
            "-static", "-static-libgcc", "-static-libstdc++",
            str(CPP_SRC),
            "-o", str(CPP_BIN)
        ]
    else:
        cmd = [
            "g++", "-std=std=c++17", "-O3",
            str(CPP_SRC), "-lpthread",
            "-o", str(CPP_BIN)
        ]

    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print("\x1b[32mCompilation successful!\x1b[0m")
            return True
        else:
            print("\x1b[31mCompilation failed:\x1b[0m")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"\x1b[31mError running compiler: {e}\x1b[0m")
        return False


def run_legacy_fallback():
    """Fall back to standard pytest execution."""
    print("\x1b[33mFalling back to standard single-threaded pytest...\x1b[0m")
    # Resolve project virtual environment python if possible
    candidates = [
        PROJECT_ROOT / "jarvis_env" / "Scripts" / "python.exe",
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
        PROJECT_ROOT / "venv" / "Scripts" / "python.exe",
        PROJECT_ROOT / "jarvis_env" / "bin" / "python",
        PROJECT_ROOT / ".venv" / "bin" / "python",
        PROJECT_ROOT / "venv" / "bin" / "python",
    ]
    python_exe = sys.executable
    for cand in candidates:
        if cand.exists():
            python_exe = str(cand)
            break

    cmd = [python_exe, "-m", "pytest"] + sys.argv[1:]
    try:
        sys.exit(subprocess.run(cmd).returncode)
    except KeyboardInterrupt:
        sys.exit(130)


def main():
    # 1. Check if compiler is available and source file exists
    if not CPP_SRC.exists():
        run_legacy_fallback()
        return

    # 2. Check if we need to compile (binary missing or source is newer)
    needs_compile = False
    if not CPP_BIN.exists():
        needs_compile = True
    else:
        # Check modified times
        src_mtime = CPP_SRC.stat().st_mtime
        bin_mtime = CPP_BIN.stat().st_mtime
        if src_mtime > bin_mtime:
            needs_compile = True

    if needs_compile:
        if has_compiler():
            success = compile_runner()
            if not success:
                run_legacy_fallback()
                return
        else:
            print("\x1b[33mWarning: g++ compiler not found. Cannot compile C++ runner.\x1b[0m")
            run_legacy_fallback()
            return

    # 3. Run the compiled C++ test runner
    cmd = [str(CPP_BIN)] + sys.argv[1:]
    try:
        sys.exit(subprocess.run(cmd).returncode)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(f"\x1b[31mError executing test runner: {e}\x1b[0m")
        run_legacy_fallback()


if __name__ == "__main__":
    main()
