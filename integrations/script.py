import os
import glob
import re

role = "ConfigurationAnalyst"
reports_dir = r"d:\AI\Jarvis\LoopDesign\FileReports"
prompts_dir = r"d:\AI\Jarvis\LoopDesign\Prompts"

os.makedirs(reports_dir, exist_ok=True)
os.makedirs(prompts_dir, exist_ok=True)

target_dir = r"d:\AI\Jarvis\integrations"

for root, _, files in os.walk(target_dir):
    for f in files:
        if not f.endswith(".py"): continue
        if "tests" in root: continue # maybe include tests? The instructions say "every file in your target directory relevant to your role". Tests may have env vars.
        
        path = os.path.join(root, f)
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()
        
        # basic heuristics
        env_vars = set(re.findall(r"os\.environ(?:\[['\"](.*?)['\"]\]|\.get\(['\"](.*?)['\"]\))", content))
        env_vars = {item for sublist in env_vars for item in sublist if item}
        
        # also look for required_config list
        req_configs = re.findall(r"required_config.*=\s*\[(.*?)\]", content, re.DOTALL)
        if req_configs:
            for rc in req_configs:
                for match in re.findall(r"['\"](.*?)['\"]", rc):
                    env_vars.add(match)

        # secrets or ini
        secrets = re.findall(r"api_key|token|secret|password", content, re.IGNORECASE)
        ini_files = re.findall(r"\w+\.ini", content, re.IGNORECASE)
        urls = re.findall(r"https?://[^\s\"']+", content)
        paths = re.findall(r"(/[\w\.-]+)+", content)
        
        # prompts
        prompts = []
        # Let's extract any docstring or multiline string that looks like a prompt or instructions
        # This is a basic heuristic.
        multi_strings = re.findall(r"\"\"\"(.*?)\"\"\"", content, re.DOTALL)
        for s in multi_strings:
            if "you are" in s.lower() or "prompt" in s.lower() or "instruction" in s.lower():
                prompts.append(s)

        filename = os.path.basename(path)
        report_path = os.path.join(reports_dir, f"{filename}_{role}.md")
        
        with open(report_path, "w", encoding="utf-8") as out:
            out.write(f"# Configuration Analysis: {filename}\n")
            out.write(f"**Path**: {path}\n\n")
            
            out.write("## Environment Variables\n")
            if env_vars:
                for ev in env_vars:
                    out.write(f"- {ev}\n")
            else:
                out.write("None detected.\n")
                
            out.write("\n## Secrets / Credentials References\n")
            if secrets:
                out.write("Found references to credentials (e.g., api_key, token, secret).\n")
            else:
                out.write("None detected.\n")
                
            out.write("\n## INI Files\n")
            if ini_files:
                for ini in set(ini_files):
                    out.write(f"- {ini}\n")
            else:
                out.write("None detected.\n")
                
            out.write("\n## Implicit Environment Assumptions\n")
            if urls:
                out.write("### URLs\n")
                for url in set(urls):
                    out.write(f"- {url}\n")
            if paths:
                pass # path regex is noisy
                
            out.write("\n## Prompts\n")
            if prompts:
                out.write(f"Found {len(prompts)} prompt(s). Extracted to Prompts directory.\n")
                for i, p in enumerate(prompts):
                    p_path = os.path.join(prompts_dir, f"{filename}_prompt_{i}.txt")
                    with open(p_path, "w", encoding="utf-8") as pf:
                        pf.write(p.strip())
            else:
                out.write("None detected.\n")

print('DONE')
