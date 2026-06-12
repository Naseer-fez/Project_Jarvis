import ast
import os
import glob
import re
from pathlib import Path

def extract_prompts_from_ast(node, file_path, prompts_dir):
    prompts_found = []
    
    for child in ast.walk(node):
        if isinstance(child, ast.Assign):
            for target in child.targets:
                if isinstance(target, ast.Name):
                    var_name = target.id
                    if 'prompt' in var_name.lower() or 'instruct' in var_name.lower():
                        # Extract the string
                        if isinstance(child.value, ast.Constant) and isinstance(child.value.value, str):
                            prompts_found.append((var_name, child.value.value))
                        elif isinstance(child.value, ast.JoinedStr):
                            # It's an f-string, let's just get the raw source if possible
                            prompts_found.append((var_name, "f-string... (omitted)"))
    
    return prompts_found

def analyze_file(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        source = f.read()
    
    try:
        tree = ast.parse(source)
    except Exception as e:
        return f"Error parsing {file_path}: {e}", []

    dependencies = []
    schemas = []
    api_contracts = []
    config_vars = []
    assumptions = []
    
    # Extract docstrings
    module_docstring = ast.get_docstring(tree)
    if module_docstring:
        assumptions.append(f"Module Docstring: {module_docstring}")
        
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                dependencies.append(n.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module if node.module else ""
            for n in node.names:
                dependencies.append(f"{module}.{n.name}")
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id.isupper():
                        config_vars.append(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                config_vars.append(f"{node.target.id} (typed)")
        elif isinstance(node, ast.ClassDef):
            schemas.append(node.name)
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    args = [a.arg for a in item.args.args]
                    api_contracts.append(f"{node.name}.{item.name}({', '.join(args)})")
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    schemas.append(f"{node.name} attribute: {item.target.id}")
        elif isinstance(node, ast.FunctionDef):
            args = [a.arg for a in node.args.args]
            api_contracts.append(f"{node.name}({', '.join(args)})")

    # regex for comments
    comments = re.findall(r'(?m)^\s*#\s*(.*)$', source)
    for c in comments:
        if any(keyword in c.lower() for keyword in ['assume', 'todo', 'note', 'hack', 'fixme', 'warn']):
            assumptions.append(f"Comment: {c}")

    prompts = extract_prompts_from_ast(tree, file_path, "")

    # Formatting the report
    report = f"# Analysis Report for {os.path.basename(file_path)}\n\n"
    report += "## Dependencies\n" + ("\n".join(f"- {d}" for d in dependencies) if dependencies else "None") + "\n\n"
    report += "## Schemas\n" + ("\n".join(f"- {s}" for s in schemas) if schemas else "None") + "\n\n"
    report += "## API Contracts\n" + ("\n".join(f"- {a}" for a in api_contracts) if api_contracts else "None") + "\n\n"
    report += "## Configuration Variables\n" + ("\n".join(f"- {c}" for c in config_vars) if config_vars else "None") + "\n\n"
    report += "## Assumptions & Notes\n" + ("\n".join(f"- {a}" for a in assumptions) if assumptions else "None") + "\n\n"
    
    return report, prompts

def main():
    target_dir = r"d:\AI\Jarvis\core"
    reports_dir = r"d:\AI\Jarvis\LoopDesign\FileReports"
    prompts_dir = r"d:\AI\Jarvis\LoopDesign\Prompts"
    
    files_analyzed = []
    
    for root, dirs, files in os.walk(target_dir):
        if '__pycache__' in root:
            continue
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                file_path = os.path.join(root, file)
                report, prompts = analyze_file(file_path)
                
                # Write report
                base_name = os.path.splitext(file)[0]
                report_file = os.path.join(reports_dir, f"{base_name}_Documentation Analyst.md")
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(report)
                
                # Write prompts
                for p_name, p_val in prompts:
                    prompt_file = os.path.join(prompts_dir, f"{base_name}_{p_name}.txt")
                    with open(prompt_file, 'w', encoding='utf-8') as f:
                        f.write(p_val)
                        
                files_analyzed.append(file_path)

    print(f"DONE. Analyzed {len(files_analyzed)} files.")
    for idx, f in enumerate(files_analyzed):
        if idx < 10:
            print(f"- {f}")
    if len(files_analyzed) > 10:
        print(f"... and {len(files_analyzed)-10} more.")

if __name__ == '__main__':
    main()
