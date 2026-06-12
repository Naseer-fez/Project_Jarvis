import os
import ast
import re
from pathlib import Path

files_to_analyze = [
    r"d:\AI\Jarvis\core\agent\agent_loop.py",
    r"d:\AI\Jarvis\core\controller\web_search.py",
    r"d:\AI\Jarvis\core\llm\client.py",
    r"d:\AI\Jarvis\core\llm\cloud_client.py",
    r"d:\AI\Jarvis\core\llm\ollama_client.py",
    r"d:\AI\Jarvis\core\memory\context_compressor.py",
    r"d:\AI\Jarvis\core\planner\planner.py",
    r"d:\AI\Jarvis\core\profile.py",
    r"d:\AI\Jarvis\core\synthesis.py",
    r"d:\AI\Jarvis\core\tools\gui_control.py",
    r"d:\AI\Jarvis\core\tools\screen.py",
    r"d:\AI\Jarvis\core\tools\vision.py",
    r"d:\AI\Jarvis\core\tools\web_tools.py",
    r"d:\AI\Jarvis\core\voice\voice_layer.py",
    r"d:\AI\Jarvis\core\voice\voice_loop.py",
    r"d:\AI\Jarvis\core\controller\automation_manager.py",
    r"d:\AI\Jarvis\core\controller\llm_dispatcher.py",
    r"d:\AI\Jarvis\core\controller\llm_orchestrator.py",
    r"d:\AI\Jarvis\core\controller\request_rules.py",
    r"d:\AI\Jarvis\core\api_analysis.py",
    r"d:\AI\Jarvis\core\temp_analyze.py",
    r"d:\AI\Jarvis\core\voice\audio_input.py"
]

role = "Prompt Recovery Specialist"
out_reports = r"d:\AI\Jarvis\LoopDesign\FileReports"
out_prompts = r"d:\AI\Jarvis\LoopDesign\Prompts"

os.makedirs(out_reports, exist_ok=True)
os.makedirs(out_prompts, exist_ok=True)

for filepath in files_to_analyze:
    if not os.path.exists(filepath):
        continue
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        continue
    
    filename = os.path.basename(filepath)
    report_path = os.path.join(out_reports, f"{filename}_{role}.md")
    
    # Analyze imports (Dependencies)
    dependencies = []
    classes = []
    functions = []
    assignments = []
    prompts_found = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                dependencies.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                dependencies.append(node.module)
        elif isinstance(node, ast.ClassDef):
            classes.append({
                'name': node.name,
                'doc': ast.get_docstring(node),
                'methods': [m.name for m in node.body if isinstance(m, ast.FunctionDef) or isinstance(m, ast.AsyncFunctionDef)]
            })
        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            # Only module level
            functions.append({
                'name': node.name,
                'doc': ast.get_docstring(node),
                'args': [a.arg for a in node.args.args]
            })
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        val = node.value.value
                        assignments.append((target.id, val))
                        if 'prompt' in target.id.lower() or 'instruction' in target.id.lower() or 'system' in target.id.lower() or '{' in val or len(val) > 100:
                            prompts_found.append({'name': target.id, 'content': val})
                    elif isinstance(node.value, ast.JoinedStr):
                        # F-string
                        val_parts = []
                        for v in node.value.values:
                            if isinstance(v, ast.Constant):
                                val_parts.append(str(v.value))
                            elif isinstance(v, ast.FormattedValue):
                                val_parts.append("{...}")
                        val = "".join(val_parts)
                        assignments.append((target.id, "f-string: " + val))
                        prompts_found.append({'name': target.id, 'content': val})
    
    # Also look for explicit dicts/lists that might hold config
    
    # Write Prompt files
    for p in prompts_found:
        prompt_file = os.path.join(out_prompts, f"{filename}_{p['name']}.md")
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(f"### Source File: {filename}\n### Variable: {p['name']}\n\n```text\n{p['content']}\n```\n")
    
    # Write Report
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# File Report: {filename}\n")
        f.write(f"**Role**: {role}\n\n")
        
        f.write("## Dependencies\n")
        for d in set(dependencies):
            f.write(f"- {d}\n")
        f.write("\n")
        
        f.write("## Configuration Variables & Constants\n")
        for k, v in assignments:
            if len(v) < 100:
                f.write(f"- `{k}`: `{v}`\n")
            else:
                f.write(f"- `{k}`: (Too long, {len(v)} chars. Extracted to Prompts if applicable)\n")
        f.write("\n")
        
        f.write("## Schemas & API Contracts\n")
        for c in classes:
            f.write(f"### Class `{c['name']}`\n")
            if c['doc']:
                f.write(f"**Assumptions/Doc**: {c['doc']}\n")
            f.write(f"**Methods**: {', '.join(c['methods'])}\n\n")
        
        for func in functions:
            f.write(f"### Function `{func['name']}`\n")
            f.write(f"**Args**: {', '.join(func['args'])}\n")
            if func['doc']:
                f.write(f"**Assumptions/Doc**: {func['doc']}\n")
            f.write("\n")
        
        f.write("## Prompts and LLM Directives\n")
        if not prompts_found:
            f.write("No explicit prompts found in module scope.\n")
        for p in prompts_found:
            f.write(f"- Extracted `{p['name']}` to Prompts directory.\n")
            
print("Done extracting reports and prompts.")
