import ast
import glob
import os
import re

CORE_DIR = r"d:\AI\Jarvis\core"
REPORTS_DIR = r"d:\AI\Jarvis\LoopDesign\FileReports"
PROMPTS_DIR = r"d:\AI\Jarvis\LoopDesign\Prompts"

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(PROMPTS_DIR, exist_ok=True)

def is_prompt(val_str, node_name):
    name_match = False
    if node_name:
        name_lower = node_name.lower()
        if any(x in name_lower for x in ["prompt", "template", "system_msg", "instruction", "sys_msg"]):
            name_match = True
    
    # check if it is a multiline string and long enough
    is_long = len(val_str.strip()) > 50
    return name_match or (is_long and ('{' in val_str and '}' in val_str or val_str.count('\n') > 2))

def parse_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return f"Error reading {filepath}: {e}"
        
    lines = content.split("\n")
    try:
        tree = ast.parse(content, filename=filepath)
    except SyntaxError as e:
        return f"SyntaxError in {filepath}: {e}"

    report = []
    rel_path = os.path.relpath(filepath, CORE_DIR)
    report.append(f"# API Analyst Report: {rel_path}\n")
    
    # 1. Dependencies
    imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.append(f"import {n.name}" + (f" as {n.asname}" if n.asname else ""))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for n in node.names:
                imports.append(f"from {module} import {n.name}" + (f" as {n.asname}" if n.asname else ""))
    
    if imports:
        report.append("## Dependencies\n" + "\n".join(f"- `{i}`" for i in imports) + "\n")
        
    # Variables, Config, Prompts
    prompts_found = []
    config_vars = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if not targets:
                continue
            name = targets[0]
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                val = node.value.value
                if is_prompt(val, name):
                    prompts_found.append((name, val))
                elif name.isupper():
                    config_vars.append((name, repr(val)))
            elif name.isupper():
                config_vars.append((name, ast.unparse(node.value)))
                
    if config_vars:
        report.append("## Configuration Variables\n" + "\n".join(f"- `{name}` = `{val}`" for name, val in config_vars) + "\n")
        
    if prompts_found:
        report.append("## Prompts Extracted\n")
        for i, (p_name, p_val) in enumerate(prompts_found):
            safe_name = os.path.basename(filepath).replace('.py', '')
            p_file = f"{safe_name}_{p_name}.txt"
            with open(os.path.join(PROMPTS_DIR, p_file), "w", encoding="utf-8") as f:
                f.write(p_val)
            report.append(f"- `{p_name}` -> Saved to `Prompts/{p_file}`\n")
            
    # Classes, Schemas, API Contracts
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    if classes:
        report.append("## Schemas & API Contracts (Classes)\n")
        for cls in classes:
            bases = [ast.unparse(b) for b in cls.bases]
            base_str = f"({', '.join(bases)})" if bases else ""
            report.append(f"### Class `{cls.name}{base_str}`")
            doc = ast.get_docstring(cls)
            if doc:
                report.append(f"> {doc}\n")
            
            # Attributes/Fields (mostly for Pydantic/Dataclasses)
            fields = []
            for n in cls.body:
                if isinstance(n, ast.AnnAssign):
                    target = ast.unparse(n.target)
                    anno = ast.unparse(n.annotation)
                    fields.append(f"  - `{target}: {anno}`")
            if fields:
                report.append("**Fields/Schema:**\n" + "\n".join(fields) + "\n")
                
            # Methods
            methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) or isinstance(n, ast.AsyncFunctionDef)]
            if methods:
                report.append("**Methods:**")
                for m in methods:
                    decs = [f"@{ast.unparse(d)}" for d in m.decorator_list]
                    args = ast.unparse(m.args)
                    returns = f" -> {ast.unparse(m.returns)}" if m.returns else ""
                    prefix = "async def" if isinstance(m, ast.AsyncFunctionDef) else "def"
                    
                    if decs:
                        report.append(f"- {' '.join(decs)}")
                    report.append(f"- `{prefix} {m.name}({args}){returns}`")
                    mdoc = ast.get_docstring(m)
                    if mdoc:
                        report.append(f"  - *{mdoc.split(chr(10))[0]}*")
            report.append("\n")

    # Functions (Endpoints, Handlers)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) or isinstance(n, ast.AsyncFunctionDef)]
    if funcs:
        report.append("## Functions & Endpoints\n")
        for f in funcs:
            decs = [f"@{ast.unparse(d)}" for d in f.decorator_list]
            args = ast.unparse(f.args)
            returns = f" -> {ast.unparse(f.returns)}" if f.returns else ""
            prefix = "async def" if isinstance(f, ast.AsyncFunctionDef) else "def"
            if decs:
                report.append(f"### {' '.join(decs)}")
            else:
                report.append(f"### `{f.name}`")
            report.append(f"`{prefix} {f.name}({args}){returns}`")
            fdoc = ast.get_docstring(f)
            if fdoc:
                report.append(f"> {fdoc}\n")
            
    # Assumptions & Comments
    assumptions = []
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("#"):
            if any(x in line.upper() for x in ["TODO", "FIXME", "NOTE", "ASSUME", "WARNING"]):
                assumptions.append(f"Line {i+1}: {line}")
    if assumptions:
        report.append("## Assumptions & Notes\n" + "\n".join(f"- {a}" for a in assumptions) + "\n")
        
    return "\n".join(report)

if __name__ == "__main__":
    py_files = glob.glob(os.path.join(CORE_DIR, "**", "*.py"), recursive=True)
    count = 0
    for f in py_files:
        if "__pycache__" in f or "api_analysis.py" in f:
            continue
        report_content = parse_file(f)
        basename = os.path.basename(f)
        safe_name = f.replace(CORE_DIR, "").replace("\\", "_").replace("/", "_").strip("_")
        out_path = os.path.join(REPORTS_DIR, f"{safe_name}_API Analyst.md")
        with open(out_path, "w", encoding="utf-8") as out:
            out.write(report_content)
        count += 1
    print(f"Processed {count} files.")
