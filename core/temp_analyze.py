import os
import ast
import re

core_dir = r"d:\AI\Jarvis\core"
reports_dir = r"d:\AI\Jarvis\LoopDesign\FileReports"
prompts_dir = r"d:\AI\Jarvis\LoopDesign\Prompts"

os.makedirs(reports_dir, exist_ok=True)
os.makedirs(prompts_dir, exist_ok=True)

for root, _, files in os.walk(core_dir):
    if '__pycache__' in root:
        continue
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            rel_path = os.path.relpath(filepath, core_dir)
            filename = file
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Use AST to find imports
            imports = []
            service_calls = set()
            exec_links = set()
            prompts = []
            
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(f"import {alias.name}")
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ''
                        for alias in node.names:
                            imports.append(f"from {module} import {alias.name}")
                    elif isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Attribute):
                            if isinstance(node.func.value, ast.Name):
                                call_name = f"{node.func.value.id}.{node.func.attr}"
                                if node.func.value.id in ['requests', 'httpx', 'aiohttp', 'urllib']:
                                    service_calls.add(call_name)
                                if node.func.value.id in ['subprocess', 'os'] and node.func.attr in ['run', 'Popen', 'system', 'execve', 'spawn']:
                                    exec_links.add(call_name)
                        elif isinstance(node.func, ast.Name):
                            if node.func.id in ['exec', 'eval']:
                                exec_links.add(node.func.id)
                    
                    # Heuristics for prompts in assignments
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and 'prompt' in target.id.lower():
                                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                                    prompts.append((target.id, node.value.value))
                    
            except Exception as e:
                pass
                
            # Regex for URLs
            urls = re.findall(r'https?://[^\s\"\'\]+', content)
            for u in urls:
                service_calls.add(f"URL: {u}")
                
            # Regex for implicit execution
            # Write Report
            if not imports and not service_calls and not exec_links and not prompts:
                continue
                
            report_path = os.path.join(reports_dir, f"{filename}_DependencyAnalyst.md")
            with open(report_path, 'w', encoding='utf-8') as rf:
                rf.write(f"# Dependency Analysis Report for {rel_path}\n\n")
                rf.write("## Library Requirements\n")
                for imp in imports:
                    rf.write(f"- {imp}\n")
                rf.write("\n## Service Dependencies\n")
                for sc in service_calls:
                    rf.write(f"- {sc}\n")
                rf.write("\n## Hidden Execution Links\n")
                for el in exec_links:
                    rf.write(f"- {el}\n")
                rf.write("\n## Configurations / Assumptions\n")
                rf.write("- Analyzed via AST and regex.\n")
                
            for idx, (p_name, p_val) in enumerate(prompts):
                p_path = os.path.join(prompts_dir, f"{filename}_{p_name}_{idx}.txt")
                with open(p_path, 'w', encoding='utf-8') as pf:
                    pf.write(p_val)
                    
print('DONE_SCRIPT')
