
"""
finder.py  —  Git & Loose-Code Scanner for Windows
Author  : Naseer-fez
Purpose : Scan relative folder, copy files to to_commit_pool, push to GitHub
Usage   : python finder.py [extra_path1] [extra_path2] ...
"""

# ══════════════════════════════════════════════════════════════════════
#  ✏️  USER CONFIGURATION  —  Edit this section to customise behaviour
# ══════════════════════════════════════════════════════════════════════

# Where to start scanning.
# Options:
#   "relative"  → scan from the folder where finder.py lives        ← DEFAULT
#   "desktop"   → C:\Users\<you>\Desktop
#   "documents" → C:\Users\<you>\Documents
#   "downloads" → C:\Users\<you>\Downloads
#   "home"      → C:\Users\<you>  (entire home folder — slow!)
#   "all"       → Desktop + Documents + Downloads
#   "custom"    → only the paths listed in CUSTOM_PATHS below
SCAN_MODE = "relative"

# Extra folders to ALWAYS include on top of SCAN_MODE (optional).
# Use raw strings r"C:\My Projects" or forward slashes "D:/Work".
# Leave as [] if you don't need extras.
CUSTOM_PATHS = []

# ══════════════════════════════════════════════════════════════════════
#  END OF USER CONFIGURATION  —  Do not edit below unless you know why
# ══════════════════════════════════════════════════════════════════════

import os
import sys
import shutil
import subprocess
import getpass
from pathlib import Path
from datetime import datetime

# ──────────────────────────────────────────────
# ANSI colour helpers
# ──────────────────────────────────────────────
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    DIM     = "\033[2m"

def enable_ansi():
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

def header(text):
    width = 70
    print(f"\n{C.BOLD}{C.CYAN}{'═' * width}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  {text}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'═' * width}{C.RESET}")

def section(text):
    print(f"\n{C.BOLD}{C.BLUE}▶ {text}{C.RESET}")

def ok(text):
    print(f"  {C.GREEN}✔ {text}{C.RESET}")

def warn(text):
    print(f"  {C.YELLOW}⚠ {text}{C.RESET}")

def err(text):
    print(f"  {C.RED}✖ {text}{C.RESET}")

# ──────────────────────────────────────────────
# SKIP / EXTENSION CONFIG
# ──────────────────────────────────────────────
SKIP_DIRS = {
    "node_modules", ".git", "vendor", "__pycache__",
    "dist", "build", ".venv", "venv", ".tox", ".mypy_cache",
    ".pytest_cache", "coverage", ".next", "out", "target",
    "to_commit_pool",   # never scan the pool folder itself
}

LOOSE_EXTENSIONS = {
    ".js", ".ts", ".jsx", ".tsx", ".py", ".java", ".cpp", ".c",
    ".h", ".hpp", ".html", ".css", ".scss", ".sass", ".go",
    ".rb", ".php", ".rs", ".swift", ".kt", ".cs", ".lua",
}

# Script's own directory
SCRIPT_DIR = Path(__file__).resolve().parent

# Pool always lives next to finder.py
POOL_DIR = SCRIPT_DIR / "to_commit_pool"

USERNAME   = os.environ.get("GITHUB_USERNAME") or getpass.getuser()
GITHUB_PAT = os.environ.get("GITHUB_PAT", "")

# ──────────────────────────────────────────────
# RESOLVE SCAN ROOTS
# ──────────────────────────────────────────────
def get_win_user():
    return (
        os.environ.get("USERNAME")
        or os.environ.get("USERPROFILE", "").split("\\")[-1]
        or getpass.getuser()
    )

def get_scan_roots(extra_args):
    win_user = get_win_user()
    base     = Path(f"C:/Users/{win_user}")

    mode_map = {
        "relative":  [SCRIPT_DIR],
        "desktop":   [base / "Desktop"],
        "documents": [base / "Documents"],
        "downloads": [base / "Downloads"],
        "home":      [base],
        "all":       [base / "Desktop", base / "Documents", base / "Downloads"],
        "custom":    [],
    }

    roots = list(mode_map.get(SCAN_MODE.lower(), [SCRIPT_DIR]))
    for p in CUSTOM_PATHS:
        roots.append(Path(p))
    for p in extra_args:
        roots.append(Path(p))

    seen, result = set(), []
    for r in roots:
        r = r.resolve()
        if r not in seen and r.exists():
            seen.add(r)
            result.append(r)
    return result

# ──────────────────────────────────────────────
# GIT HELPERS
# ──────────────────────────────────────────────
def run_git(args, cwd):
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except FileNotFoundError:
        return "", "git not found", 1
    except Exception as e:
        return "", str(e), 1

def get_repo_status(repo_path: Path):
    status = {
        "modified":   [],
        "staged":     [],
        "untracked":  [],
        "unpushed":   [],
        "branch":     "unknown",
        "remote":     None,
        "has_remote": False,
    }

    branch_out, _, _ = run_git(["branch", "--show-current"], repo_path)
    status["branch"] = branch_out or "HEAD (detached)"

    out, _, code = run_git(["status", "--porcelain=v1"], repo_path)
    if code == 0:
        for line in out.splitlines():
            if len(line) < 2:
                continue
            xy   = line[:2]
            path = line[3:].strip().strip('"')
            if xy[0] in ("M", "A", "D", "R", "C") and xy[0] != " ":
                status["staged"].append(path)
            if xy[1] in ("M", "D"):
                status["modified"].append(path)
            if xy == "??":
                status["untracked"].append(path)

    remote_out, _, _ = run_git(["remote", "get-url", "origin"], repo_path)
    if remote_out:
        status["remote"]     = remote_out
        status["has_remote"] = True

    if status["has_remote"]:
        run_git(["fetch", "--quiet"], repo_path)
        log_out, _, log_code = run_git(
            ["log", f"origin/{status['branch']}..HEAD", "--oneline"], repo_path
        )
        if log_code == 0 and log_out:
            status["unpushed"] = log_out.splitlines()
    else:
        log_out, _, log_code = run_git(["log", "--oneline"], repo_path)
        if log_code == 0 and log_out:
            status["unpushed"] = log_out.splitlines()

    return status

# ──────────────────────────────────────────────
# DIRECTORY WALKER
# ──────────────────────────────────────────────
def walk(root: Path):
    try:
        entries = list(root.iterdir())
    except PermissionError:
        return

    if (root / ".git").exists():
        yield root, True
        return

    for entry in entries:
        if not entry.is_dir():
            continue
        if entry.name in SKIP_DIRS:
            continue
        yield from walk(entry)

def find_loose_files(root: Path, git_repo_paths: set):
    loose = []

    def _walk(path: Path):
        try:
            entries = list(path.iterdir())
        except PermissionError:
            return
        for entry in entries:
            if entry.is_dir():
                if entry.name in SKIP_DIRS:
                    continue
                if any(str(entry).startswith(str(gr)) for gr in git_repo_paths):
                    continue
                _walk(entry)
            elif entry.is_file():
                if entry.suffix.lower() in LOOSE_EXTENSIONS:
                    if not any(str(entry).startswith(str(gr)) for gr in git_repo_paths):
                        loose.append(entry)

    _walk(root)
    return loose

# ──────────────────────────────────────────────
# COPY TO POOL
# ──────────────────────────────────────────────
def copy_to_pool(src: Path, pool_root: Path, relative_base: Path):
    try:
        rel = src.relative_to(relative_base)
    except ValueError:
        rel = Path(src.name)

    project_name = relative_base.name
    dest = pool_root / project_name / rel
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        if src.stat().st_mtime <= dest.stat().st_mtime:
            return False

    shutil.copy2(str(src), str(dest))
    return True

# ──────────────────────────────────────────────
# GIT PUSH POOL TO GITHUB
# ──────────────────────────────────────────────
def push_pool_to_github(pool_root: Path, repo_root: Path, copied: int):
    section("Pushing to_commit_pool → GitHub…")

    if copied == 0:
        warn("No new files to push — skipping git push.")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Stage the pool folder
    _, _, code = run_git(["add", str(pool_root.relative_to(repo_root))], repo_root)
    if code != 0:
        # Try staging with full path
        run_git(["add", "to_commit_pool/"], repo_root)

    # Check if there's anything to commit
    diff_out, _, _ = run_git(["diff", "--staged", "--name-only"], repo_root)
    if not diff_out:
        ok("Nothing new to commit — pool already up to date on GitHub.")
        return

    # Commit
    msg = f"chore: update commit pool ({copied} files) — {timestamp}"
    _, commit_err, commit_code = run_git(["commit", "-m", msg], repo_root)
    if commit_code != 0:
        err(f"Git commit failed: {commit_err}")
        return
    ok(f"Committed: {msg}")

    # Push
    _, push_err, push_code = run_git(["push", "origin", "HEAD"], repo_root)
    if push_code != 0:
        err(f"Git push failed: {push_err}")
        warn("Try running manually: git push origin main")
    else:
        ok("✅ Pushed to GitHub successfully!")

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    enable_ansi()

    extra_args = sys.argv[1:]
    scan_roots = get_scan_roots(extra_args)
    pool_root  = POOL_DIR

    # ── Banner ──────────────────────────────────
    header(f"Git & Loose-Code Scanner  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  {C.MAGENTA}GitHub user : {C.BOLD}{USERNAME}{C.RESET}")
    print(f"  {C.MAGENTA}PAT loaded  : {C.BOLD}{'YES ✔' if GITHUB_PAT else 'NO — set GITHUB_PAT env var'}{C.RESET}")
    print(f"  {C.MAGENTA}Scan mode   : {C.BOLD}{SCAN_MODE.upper()}{C.RESET}")
    print(f"  {C.MAGENTA}Script dir  : {C.BOLD}{SCRIPT_DIR}{C.RESET}")
    print(f"  {C.MAGENTA}Pool folder : {C.BOLD}{pool_root}{C.RESET}")

    section("Scan roots")
    for r in scan_roots:
        ok(str(r))

    # ── Discover repos ───────────────────────────
    section("Discovering git repositories…")
    git_repos = []
    for root in scan_roots:
        for repo_path, _ in walk(root):
            git_repos.append(repo_path)

    print(f"  Found {C.BOLD}{C.GREEN}{len(git_repos)}{C.RESET} git repositories.\n")

    # ── Analyse each repo ────────────────────────
    all_dirty_files = []
    total_modified  = 0
    total_staged    = 0
    total_untracked = 0
    total_unpushed  = 0
    git_repo_paths  = set(git_repos)

    for repo in git_repos:
        st    = get_repo_status(repo)
        dirty = st["modified"] or st["staged"] or st["untracked"] or st["unpushed"]
        if not dirty:
            continue

        branch_col = C.GREEN if st["has_remote"] else C.YELLOW
        print(f"\n  {C.BOLD}{C.WHITE}{repo}{C.RESET}  "
              f"{branch_col}[{st['branch']}]{C.RESET}  "
              f"{C.DIM}{'→ ' + st['remote'] if st['remote'] else '(no remote)'}{C.RESET}")

        def _report(label, colour, items):
            if items:
                print(f"    {colour}{label} ({len(items)}):{C.RESET}")
                for f in items[:10]:
                    print(f"      {C.DIM}{f}{C.RESET}")
                if len(items) > 10:
                    print(f"      {C.DIM}… and {len(items)-10} more{C.RESET}")

        _report("Modified (unstaged)",  C.YELLOW,  st["modified"])
        _report("Staged (uncommitted)", C.MAGENTA, st["staged"])
        _report("Untracked",            C.RED,     st["untracked"])
        _report("Unpushed commits",     C.CYAN,    st["unpushed"])

        total_modified  += len(st["modified"])
        total_staged    += len(st["staged"])
        total_untracked += len(st["untracked"])
        total_unpushed  += len(st["unpushed"])

        for rel_f in (st["modified"] + st["staged"] + st["untracked"]):
            abs_f = repo / rel_f
            if abs_f.exists() and abs_f.is_file():
                all_dirty_files.append((abs_f, repo))

    # ── Loose files ──────────────────────────────
    section("Scanning for loose code files (outside any git repo)…")
    all_loose = []
    for root in scan_roots:
        loose = find_loose_files(root, git_repo_paths)
        all_loose.extend([(f, root) for f in loose])

    if all_loose:
        print(f"  Found {C.BOLD}{C.RED}{len(all_loose)}{C.RESET} loose code files:\n")
        for f, base in all_loose[:30]:
            print(f"    {C.RED}{f}{C.RESET}")
        if len(all_loose) > 30:
            print(f"    {C.DIM}… and {len(all_loose)-30} more{C.RESET}")
    else:
        ok("No loose code files found outside git repos.")

    # ── Copy to pool ─────────────────────────────
    section(f"Copying files → {pool_root}")
    pool_root.mkdir(parents=True, exist_ok=True)

    copied  = 0
    skipped = 0

    for abs_f, base in all_dirty_files + all_loose:
        if copy_to_pool(abs_f, pool_root, base):
            copied += 1
        else:
            skipped += 1

    ok(f"{copied} files copied/updated, {skipped} already up-to-date.")

    # ── Manifest ─────────────────────────────────
    manifest_path = pool_root / "_MANIFEST.txt"
    with open(manifest_path, "w", encoding="utf-8") as mf:
        mf.write(f"# to_commit_pool manifest — generated {datetime.now().isoformat()}\n")
        mf.write(f"# GitHub user : {USERNAME}\n")
        mf.write(f"# Scan mode   : {SCAN_MODE}\n")
        mf.write(f"# Script dir  : {SCRIPT_DIR}\n\n")
        mf.write("## Dirty git files\n")
        for abs_f, _ in all_dirty_files:
            mf.write(f"{abs_f}\n")
        mf.write("\n## Loose code files\n")
        for abs_f, _ in all_loose:
            mf.write(f"{abs_f}\n")

    # ── Auto push to GitHub ───────────────────────
    push_pool_to_github(pool_root, SCRIPT_DIR, copied)

    # ── Summary ──────────────────────────────────
    header("SUMMARY")
    rows = [
        ("Git repos scanned",          len(git_repos),   C.CYAN),
        ("Modified (unstaged) files",  total_modified,   C.YELLOW),
        ("Staged (uncommitted) files", total_staged,     C.MAGENTA),
        ("Untracked files",            total_untracked,  C.RED),
        ("Unpushed commit messages",   total_unpushed,   C.CYAN),
        ("Loose code files",           len(all_loose),   C.RED),
        ("Files copied to pool",       copied,           C.GREEN),
        ("Files already up-to-date",   skipped,          C.DIM),
    ]
    for label, count, colour in rows:
        bar = "█" * min(count, 40)
        print(f"  {colour}{label:<30} {C.BOLD}{count:>5}  {bar}{C.RESET}")

    total_action = total_modified + total_staged + total_untracked + len(all_loose)
    print(f"\n  {C.BOLD}{C.WHITE}Total files needing attention : {total_action}{C.RESET}")
    print(f"  {C.BOLD}{C.GREEN}Pool location : {pool_root}{C.RESET}")
    print(f"  {C.DIM}Manifest      : {manifest_path}{C.RESET}\n")

    if total_action == 0:
        print(f"  {C.GREEN}{C.BOLD}🎉 Everything is clean! Nothing to commit.{C.RESET}\n")
    else:
        print(f"  {C.YELLOW}{C.BOLD}⚡ Pool pushed to GitHub — workflow will commit 1 file/day automatically.{C.RESET}\n")

if __name__ == "__main__":
    main()