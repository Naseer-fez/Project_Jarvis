import os
import queue
import threading
import fnmatch
import asyncio
from pathlib import Path


# ----------------------------------------------------------------------
# Fast Python Multi-threaded Traversal & Grep Fallback Engine
# ----------------------------------------------------------------------
class PythonSearchEngine:
    def __init__(self, start_paths, query=None, content_query=None, num_threads=16, case_sensitive=False, no_skip=False, max_results=2000):
        self.start_paths = [Path(p) for p in start_paths]
        self.query = query
        self.content_query = content_query
        self.num_threads = num_threads
        self.case_sensitive = case_sensitive
        self.no_skip = no_skip
        self.max_results = max_results
        
        self.q = queue.Queue()
        self.active_workers = 0
        self.active_workers_lock = threading.Lock()
        self.results = []
        self.results_lock = threading.Lock()
        self.done_event = threading.Event()
        
        self.files_scanned = 0
        self.dirs_scanned = 0
        
        self.skip_dirs = {
            "$recycle.bin",
            "system volume information",
            "node_modules",
            ".git",
            ".venv",
            "venv",
            "appdata",
            "winsxs",
            "servicing",
            "windows\\temp",
            "microsoft",
            "recovery"
        }

    def should_skip(self, path):
        if self.no_skip:
            return False
        p_lower = str(path).lower()
        for skip in self.skip_dirs:
            if skip in p_lower:
                return True
        return False

    def is_binary(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                chunk = f.read(1024)
                return b'\x00' in chunk
        except Exception:
            return True

    def search_file_content(self, filepath):
        if not self.content_query:
            return
        try:
            if os.path.getsize(filepath) > 20 * 1024 * 1024:  # 20MB limit
                return
            if self.is_binary(filepath):
                return
            
            target = self.content_query if self.case_sensitive else self.content_query.lower()
            # Open file with error-handling to prevent decode failures from stopping walk
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    search_line = line if self.case_sensitive else line.lower()
                    if target in search_line:
                        with self.results_lock:
                            if len(self.results) < self.max_results:
                                self.results.append({
                                    "type": "match",
                                    "path": str(filepath),
                                    "line": line_num,
                                    "text": line.strip()
                                })
                            else:
                                self.done_event.set()
                                return
        except Exception:
            pass

    def worker(self):
        while not self.done_event.is_set():
            try:
                # Short timeout to periodically check done_event
                current_dir = self.q.get(timeout=0.05)
            except queue.Empty:
                with self.active_workers_lock:
                    if self.active_workers == 0:
                        self.done_event.set()
                        break
                continue
            
            with self.active_workers_lock:
                self.active_workers += 1
            
            self.dirs_scanned += 1
            try:
                with os.scandir(current_dir) as it:
                    for entry in it:
                        if self.done_event.is_set():
                            break
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                if not self.should_skip(entry.path):
                                    self.q.put(entry.path)
                            elif entry.is_file(follow_symlinks=False):
                                self.files_scanned += 1
                                filename = entry.name
                                match = False
                                if not self.query:
                                    match = True
                                else:
                                    if self.case_sensitive:
                                        match = fnmatch.fnmatch(filename, self.query)
                                    else:
                                        match = fnmatch.fnmatch(filename.lower(), self.query.lower())
                                
                                if match:
                                    if self.content_query:
                                        self.search_file_content(entry.path)
                                    else:
                                        with self.results_lock:
                                            if len(self.results) < self.max_results:
                                                self.results.append({
                                                    "type": "file",
                                                    "path": str(entry.path)
                                                })
                                            else:
                                                self.done_event.set()
                                                break
                        except Exception:
                            pass
            except Exception:
                pass
            finally:
                with self.active_workers_lock:
                    self.active_workers -= 1
                self.q.task_done()

    def run(self):
        # Initialize queue with root paths
        for path in self.start_paths:
            if path.exists():
                if path.is_dir():
                    self.q.put(str(path))
                else:
                    self.files_scanned += 1
                    filename = path.name
                    match = False
                    if not self.query:
                        match = True
                    else:
                        if self.case_sensitive:
                            match = fnmatch.fnmatch(filename, self.query)
                        else:
                            match = fnmatch.fnmatch(filename.lower(), self.query.lower())
                    if match:
                        if self.content_query:
                            self.search_file_content(path)
                        else:
                            self.results.append({
                                "type": "file",
                                "path": str(path)
                            })
        
        threads = []
        for _ in range(self.num_threads):
            t = threading.Thread(target=self.worker)
            t.start()
            threads.append(t)
            
        for t in threads:
            t.join()
            
        return {
            "results": self.results,
            "files_scanned": self.files_scanned,
            "dirs_scanned": self.dirs_scanned,
            "matches_count": len(self.results)
        }

# ----------------------------------------------------------------------
# Helper: Get all Windows Logical Drives
# ----------------------------------------------------------------------
def get_windows_drives():
    import ctypes
    drives = []
    bitmask = ctypes.windll.kernel32.GetLogicalDrives()
    for letter in range(26):
        if bitmask & (1 << letter):
            drive = f"{chr(65 + letter)}:\\"
            drives.append(drive)
    return drives

# ----------------------------------------------------------------------
# Fast Search Runner (Subprocess for C++ or Fallback to Python)
# ----------------------------------------------------------------------
async def run_fast_search(path="all", query="", content="", threads=8, case_sensitive=False, no_skip=False, max_results=1000):
    """
    Search files by name pattern and/or by file content (grep) using C++ executable.
    If the executable is not compiled or fails to run, it falls back to a high-performance Python threaded crawler.
    """
    # 1. Determine roots
    if path == "all":
        roots = get_windows_drives() if os.name == "nt" else ["/"]
    else:
        roots = [path]

    # Resolve binary path
    project_root = Path(__file__).resolve().parent.parent.parent
    exe_path = project_root / "core" / "tools" / "fast_search" / "fast_search.exe"
    
    # Try using the compiled C++ binary first
    if exe_path.exists():
        args = []
        for r in roots:
            args.extend(["--path", str(r)])
        if query:
            args.extend(["--query", query])
        if content:
            args.extend(["--content", content])
        args.extend(["--threads", str(threads)])
        args.extend(["--max-results", str(max_results)])
        if case_sensitive:
            args.append("--case")
        if no_skip:
            args.append("--no-skip")

        try:
            process = await asyncio.create_subprocess_exec(
                str(exe_path),
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode == 0:
                import re
                ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                output_lines = stdout.decode('utf-8', errors='replace').splitlines()
                results = []
                summary = {}
                for line in output_lines:
                    line_clean = ansi_escape.sub('', line)
                    if line_clean.startswith("[FILE] "):
                        results.append({
                            "type": "file",
                            "path": line_clean[7:].strip()
                        })
                    elif line_clean.startswith("[MATCH] "):
                        # Format: [MATCH] path:line: content
                        parts = line_clean[8:].split(':', 2)
                        if len(parts) >= 3:
                            results.append({
                                "type": "match",
                                "path": parts[0].strip(),
                                "line": int(parts[1].strip()) if parts[1].strip().isdigit() else parts[1],
                                "text": parts[2].strip()
                            })
                    elif "Files scanned" in line_clean:
                        summary["files_scanned"] = line_clean.split(":")[-1].strip()
                    elif "Folders scanned" in line_clean:
                        summary["dirs_scanned"] = line_clean.split(":")[-1].strip()
                    elif "Elapsed time" in line_clean:
                        summary["elapsed"] = line_clean.split(":")[-1].strip()

                return {
                    "engine": "cpp",
                    "results": results,
                    "summary": summary
                }
        except Exception:
            # Fall back silently to Python if process launch fails
            pass

    # 2. Python fallback engine
    engine = PythonSearchEngine(
        start_paths=roots,
        query=query if query else None,
        content_query=content if content else None,
        num_threads=threads * 2, # Spin more threads for Python to hide I/O latency
        case_sensitive=case_sensitive,
        no_skip=no_skip,
        max_results=max_results
    )
    
    start_time = asyncio.get_event_loop().time()
    # Run in standard executor to avoid blocking the asyncio event loop
    loop = asyncio.get_running_loop()
    res = await loop.run_in_executor(None, engine.run)
    elapsed = asyncio.get_event_loop().time() - start_time
    
    res["engine"] = "python"
    res["summary"] = {
        "files_scanned": res["files_scanned"],
        "dirs_scanned": res["dirs_scanned"],
        "elapsed": f"{elapsed:.4f} seconds"
    }
    return res

if __name__ == "__main__":
    # Test script locally
    async def test():
        print("Testing fast search Python/C++ tool...")
        res = await run_fast_search(path=".", query="*.py", threads=4)
        print(f"Engine used: {res['engine']}")
        print(f"Summary: {res['summary']}")
        print(f"Results Count: {len(res['results'])}")
        print("First 3 results:")
        for r in res['results'][:3]:
            print(r)
    asyncio.run(test())
