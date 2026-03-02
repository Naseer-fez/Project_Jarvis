import os
import subprocess
import glob

def run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)

print("=== JARVIS FULL VERIFICATION SUITE ===\n")

# 1. No hardcoded Windows paths
print("--- [1/8] No hardcoded Windows paths ---")
fail_1 = False
count_1 = 0
for root, _, files in os.walk("core"):
    for file in files:
        if file.endswith(".py") and file != "__pycache__":
            filepath = os.path.join(root, file)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                if "D:/AI/Jarvis" in content or "D:\\AI\\Jarvis" in content or "D:\\\\AI\\\\Jarvis" in content:
                    fail_1 = True
                    count_1 += 1
if not fail_1:
    print("PASS")
else:
    print(f"FAIL: {count_1} hardcoded paths remain")

# 2. Correct LLMClientV2 imported in controller
print("\n--- [2/8] Correct LLMClientV2 imported in controller ---")
with open("core/controller_v2.py", 'r', encoding='utf-8') as f:
    if "from core.llm.client import LLMClientV2" in f.read():
        print("PASS")
    else:
        print("FAIL: wrong import")

# 3. No asyncio.new_event_loop in production async code
print("\n--- [3/8] No asyncio.new_event_loop in production async code ---")
count_3 = 0
for file in ["core/llm/client.py", "core/planning/intents.py"]:
    if os.path.exists(file):
        with open(file, 'r', encoding='utf-8') as f:
            count_3 += f.read().count("asyncio.new_event_loop")
if count_3 == 0:
    print("PASS")
else:
    print(f"WARN: {count_3} occurrences remain — verify they use ThreadPoolExecutor")

# 4. settings.env not in git
print("\n--- [4/8] settings.env not in git ---")
res_4 = run("git ls-files config/settings.env")
if res_4.stdout.strip():
    print("FAIL: settings.env is tracked")
else:
    print("PASS")

# 5. No .exe binaries in git
print("\n--- [5/8] No .exe binaries in git ---")
res_5 = run("git ls-files *.exe")
if res_5.stdout.strip():
    print("FAIL: .exe files tracked")
else:
    print("PASS")

# 6. pytest.ini has norecursedirs
print("\n--- [6/8] pytest.ini has norecursedirs ---")
if os.path.exists("pytest.ini"):
    with open("pytest.ini", 'r', encoding='utf-8') as f:
        content = f.read()
        if "norecursedirs" in content and "Failed" in content:
            print("PASS")
        else:
            print("FAIL: norecursedirs not configured")
else:
    print("FAIL: pytest.ini not found")

# 7. icalendar installed
print("\n--- [7/8] icalendar installed ---")
try:
    import icalendar
    print("PASS")
except ImportError:
    print("FAIL: run pip install icalendar python-dateutil")

# 8. Full test suite
print("\n--- [8/8] Full test suite ---")
res_8 = run("pytest tests/ -q --timeout=30 --tb=line")
lines = res_8.stdout.strip().split('\n')
for line in lines[-5:]:
    print(line)
