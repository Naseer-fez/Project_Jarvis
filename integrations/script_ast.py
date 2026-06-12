import os
import ast
import json

role = "Configuration Analyst"
reports_dir = r"d:\AI\Jarvis\LoopDesign\FileReports"
prompts_dir = r"d:\AI\Jarvis\LoopDesign\Prompts"

os.makedirs(reports_dir, exist_ok=True)
os.makedirs(prompts_dir, exist_ok=True)

target_dir = r"d:\AI\Jarvis\integrations"

class Analyzer(ast.NodeVisitor):
    def __init__(self):
        self.dependencies = []
        self.env_vars = []
        self.tools = []
        self.config_vars = []
        self.prompts = []
        self.assumptions = []

    def visit_Import(self, node):
        for alias in node.names:
            self.dependencies.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            self.dependencies.append(node.module)
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'get':
            if isinstance(node.func.value, ast.Attribute) and node.func.value.attr == 'environ':
                if node.args and isinstance(node.args[0], ast.Constant):
                    self.env_vars.append(node.args[0].value)
        self.generic_visit(node)

    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Attribute) and node.value.attr == 'environ':
            if isinstance(node.slice, ast.Constant):
                self.env_vars.append(node.slice.value)
        self.generic_visit(node)

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == 'required_config':
                if isinstance(node.value, ast.List):
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Constant):
                            self.config_vars.append(elt.value)
        self.generic_visit(node)

    def visit_Dict(self, node):
        # Extremely basic heuristic for tool schemas
        keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
        if 'name' in keys and 'description' in keys and 'args' in keys:
            # likely a tool schema, we try to reconstruct it or just note it
            name_idx = keys.index('name')
            if isinstance(node.values[name_idx], ast.Constant):
                self.tools.append(node.values[name_idx].value)
        self.generic_visit(node)
        
    def visit_ClassDef(self, node):
        # docstrings can be prompts
        doc = ast.get_docstring(node)
        if doc and ('prompt' in doc.lower() or 'instruction' in doc.lower() or 'you are' in doc.lower()):
            self.prompts.append(doc)
        self.generic_visit(node)
        
    def visit_FunctionDef(self, node):
        doc = ast.get_docstring(node)
        if doc and ('prompt' in doc.lower() or 'instruction' in doc.lower() or 'you are' in doc.lower()):
            self.prompts.append(doc)
        self.generic_visit(node)


for root, _, files in os.walk(target_dir):
    for f in files:
        if not f.endswith(".py"): continue
        path = os.path.join(root, f)
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()
            
        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue
            
        analyzer = Analyzer()
        analyzer.visit(tree)
        
        # Additional manual prompt search
        import re
        multi_strings = re.findall(r'\"\"\"(.*?)\"\"\"', content, re.DOTALL)
        for s in multi_strings:
            if ("you are" in s.lower() or "prompt" in s.lower() or "instruction" in s.lower()) and s not in analyzer.prompts:
                analyzer.prompts.append(s)
                
        # Secrets search
        secrets = re.findall(r'api_key|token|secret|password', content, re.IGNORECASE)
        urls = re.findall(r'https?://[^\s\"\']+', content)

        filename = os.path.basename(path)
        report_path = os.path.join(reports_dir, f"{filename}_{role}.md")
        
        with open(report_path, "w", encoding="utf-8") as out:
            out.write(f"# Configuration Analysis: {filename}\n")
            out.write(f"**Path**: {path}\n\n")
            
            out.write("## 1. Environment & Configuration Variables\n")
            all_vars = set(analyzer.env_vars + analyzer.config_vars)
            if all_vars:
                for ev in sorted(all_vars):
                    out.write(f"- {ev}\n")
            else:
                out.write("None detected.\n")
                
            out.write("\n## 2. Secrets & Credentials\n")
            if secrets:
                out.write(f"Detected potential secret references: {', '.join(set([s.lower() for s in secrets]))}\n")
            else:
                out.write("None detected.\n")
                
            out.write("\n## 3. Dependencies\n")
            if analyzer.dependencies:
                for dep in sorted(set(analyzer.dependencies)):
                    out.write(f"- {dep}\n")
            else:
                out.write("None detected.\n")
                
            out.write("\n## 4. API Contracts & Tools (Schemas)\n")
            if analyzer.tools:
                for t in sorted(set(analyzer.tools)):
                    out.write(f"- Tool Schema: {t}\n")
            else:
                out.write("None explicitly defined via standard dict format.\n")
                
            out.write("\n## 5. Implicit Assumptions (URLs, hardcoded paths)\n")
            if urls:
                out.write("### URLs\n")
                for url in sorted(set(urls)):
                    out.write(f"- {url}\n")
            else:
                out.write("No hardcoded URLs detected.\n")
                
            out.write("\n## 6. Prompts\n")
            if analyzer.prompts:
                out.write(f"Found {len(analyzer.prompts)} prompt(s). Extracted to Prompts directory.\n")
                for i, p in enumerate(analyzer.prompts):
                    p_path = os.path.join(prompts_dir, f"{filename}_prompt_{i}.txt")
                    with open(p_path, "w", encoding="utf-8") as pf:
                        pf.write(p.strip())
            else:
                out.write("None detected.\n")

print('DONE_AST')
