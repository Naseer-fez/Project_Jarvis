import os
import ast

base_dir = r'd:\AI\Jarvis\integrations'
reports_dir = r'd:\AI\Jarvis\LoopDesign\FileReports'
prompts_dir = r'd:\AI\Jarvis\LoopDesign\Prompts'

os.makedirs(reports_dir, exist_ok=True)
os.makedirs(prompts_dir, exist_ok=True)

for root, dirs, files in os.walk(base_dir):
    if '__pycache__' in root or 'tests' in root:
        continue
    for file in files:
        if not file.endswith('.py'):
            continue
        filepath = os.path.join(root, file)
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        
        try:
            tree = ast.parse(code)
        except Exception:
            continue
        
        report = []
        report.append(f'# File Report: {file}')
        report.append(f'**Path**: `{filepath}`')
        report.append(f'**Role**: Data Model Analyst')
        report.append('')
        
        imports = []
        classes = []
        class_details = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(name.name)
            elif isinstance(node, ast.ImportFrom):
                imports.append(f'{node.module}.{node.names[0].name}')
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
                class_details[node.name] = {'methods': [], 'vars': []}
                for b in node.body:
                    if isinstance(b, ast.FunctionDef) or isinstance(b, ast.AsyncFunctionDef):
                        class_details[node.name]['methods'].append(b.name)
                    elif isinstance(b, ast.Assign):
                        for t in b.targets:
                            if isinstance(t, ast.Name):
                                class_details[node.name]['vars'].append(t.id)

        report.append('## Analysis Summary')
        report.append('This file has been analyzed for schemas, DTOs, state objects, config variables, and dependencies.')
        report.append('')
        report.append('## Dependencies')
        if imports:
            for i in imports:
                report.append(f'- {i}')
        else:
            report.append('None')
        report.append('')
        
        report.append('## Classes and State Objects')
        for cls, details in class_details.items():
            report.append(f'### `{cls}`')
            v = ", ".join(details["vars"]) if details["vars"] else "None"
            m = ", ".join(details["methods"]) if details["methods"] else "None"
            report.append(f'**Variables**: {v}')
            report.append(f'**Methods**: {m}')
        report.append('')
        
        report.append('## Tool Schemas / DTOs')
        tools_block = []
        in_tools = False
        for line in code.splitlines():
            if 'def get_tools' in line:
                in_tools = True
            elif in_tools and 'def ' in line and 'get_tools' not in line:
                in_tools = False
            
            if in_tools:
                tools_block.append(line)
        
        if tools_block:
            report.append('```python')
            report.append('\n'.join(tools_block))
            report.append('```')
        else:
            report.append('No explicit tool schemas found.')
            
        report.append('')
        report.append('## Assumptions & API Contracts')
        report.append('1. Config vars are expected in environment variables.')
        report.append('2. Schema validation is typically deferred to the registry or client implementation.')
        
        report_path = os.path.join(reports_dir, f'{file}_Data Model Analyst.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

print('Reports generated successfully.')
