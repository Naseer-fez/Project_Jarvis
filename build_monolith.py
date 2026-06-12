import os
import ast
import re
from collections import defaultdict

def get_python_files(root_dir):
    py_files = []
    exclude_dirs = {
        '.git', '.mypy_cache', '.pytest_cache', '.ruff_cache', 'tests',
        'build', 'dist', 'htmlcov', 'venv', 'env', 'jarvis_env', 'min_venv', 'site-packages',
        'chroma_db', 'dashboard', 'bin', 'docs', 'data', 'logs',
        'DesignChange', 'LoopDesign', 'Cartography_Outputs', 'Report', 'Final'
    }
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for file in filenames:
            if file.endswith('.py') and not file.startswith('test_') and not file.startswith('script'):
                py_files.append(os.path.join(dirpath, file))
    return py_files

def main():
    root_dir = r"d:\AI\Jarvis"
    files = get_python_files(root_dir)
    
    # Exclude analysis scripts
    exclude_files = ['analyze_repo.py', 'analyze.py', 'api_analysis.py', 'automated_test.py', 'gen_docs.py', 'generate_cartography.py', 'qa_test.py', 'run_tests.py', 'scratch_script.py', 'test.py', 'check_duplicates.py', 'fix_cursor_leaks.py', 'build_monolith.py', 'jarvis_monolith.py', 'core/temp_analyze.py']
    files = [f for f in files if os.path.relpath(f, root_dir).replace('\\', '/') not in exclude_files]

    # 1. Parse AST and find all definitions
    defined_names = defaultdict(list)
    file_asts = {}
    file_contents = {}

    for f in files:
        rel = os.path.relpath(f, root_dir).replace('\\', '/')
        try:
            with open(f, 'r', encoding='utf-8') as file:
                content = file.read()
            tree = ast.parse(content, filename=rel)
            file_contents[rel] = content
            file_asts[rel] = tree
        except Exception as e:
            print(f"Error parsing {rel}: {e}")
            continue

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                defined_names[node.name].append(rel)
            elif isinstance(node, ast.FunctionDef):
                defined_names[node.name].append(rel)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        defined_names[target.id].append(rel)

    # 2. Identify duplicates and create unique names
    duplicates = {name: locs for name, locs in defined_names.items() if len(locs) > 1}
    # Don't rename common dunders or simple names unless necessary
    skip_renaming = {'__init__', 'main', 'logger', 'get_logger', 'run', 'ToolResult'}
    # Wait, ToolResult needs renaming! Let's NOT skip ToolResult.
    skip_renaming = {'__init__', 'main', 'logger', 'get_logger'}
    
    rename_map = {} # (module, original_name) -> new_name
    for name, locs in duplicates.items():
        if name in skip_renaming: continue
        for loc in locs:
            prefix = loc.replace('/', '_').replace('.py', '')
            new_name = f"{prefix}_{name}"
            rename_map[(loc, name)] = new_name

    # 3. Figure out what each file imports, to map usages to their renamed targets
    # file -> {original_name: new_name}
    file_renames = defaultdict(dict)
    
    # Also collect third-party/stdlib imports
    global_imports = set()
    
    # And build dependency graph for topological sort
    deps = defaultdict(set) # file -> set of files it depends on
    module_to_file = {} # "core.memory" -> "core/memory/__init__.py" or "core/memory.py"
    for rel in file_contents.keys():
        mod = rel.replace('.py', '').replace('/', '.')
        if mod.endswith('.__init__'):
            mod = mod[:-9]
        module_to_file[mod] = rel
        
    for rel, tree in file_asts.items():
        # local definitions
        for node in tree.body:
            if isinstance(node, ast.ClassDef) or isinstance(node, ast.FunctionDef):
                if (rel, node.name) in rename_map:
                    file_renames[rel][node.name] = rename_map[(rel, node.name)]
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and (rel, target.id) in rename_map:
                        file_renames[rel][target.id] = rename_map[(rel, target.id)]
                        
        # imports
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # check if it's an internal module
                    base_mod = alias.name.split('.')[0]
                    if base_mod in ['core', 'config', 'integrations', 'plugins', 'memory', 'workflows', 'runtime', 'dashboard', 'api', 'tools']:
                        # It's an internal import. Add to deps
                        for mod_name, f in module_to_file.items():
                            if mod_name == alias.name or mod_name.startswith(alias.name + '.'):
                                deps[rel].add(f)
                    else:
                        global_imports.add(f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""))
            elif isinstance(node, ast.ImportFrom):
                module = node.module if node.module else ""
                base_mod = module.split('.')[0]
                if base_mod in ['core', 'config', 'integrations', 'plugins', 'memory', 'workflows', 'runtime', 'dashboard', 'api', 'tools']:
                    # internal import
                    target_file = None
                    if module in module_to_file:
                        target_file = module_to_file[module]
                    elif f"{module}.__init__" in module_to_file:
                        target_file = module_to_file[f"{module}.__init__"]
                        
                    if target_file:
                        deps[rel].add(target_file)
                        for alias in node.names:
                            # if alias.name was renamed in target_file, we must rename it here too!
                            if (target_file, alias.name) in rename_map:
                                local_name = alias.asname or alias.name
                                file_renames[rel][local_name] = rename_map[(target_file, alias.name)]
                else:
                    if node.level > 0:
                        continue # Skip relative imports if any leak through
                    names_str = ", ".join(f"{a.name}" + (f" as {a.asname}" if a.asname else "") for a in node.names)
                    global_imports.add(f"from {module} import {names_str}")

    # 4. Topological Sort
    sorted_files = []
    visited = set()
    temp_mark = set()
    
    def visit(n):
        if n in temp_mark: return # Cycle detected, ignore to break cycle
        if n in visited: return
        temp_mark.add(n)
        for m in sorted(deps[n]): # sorted for determinism
            visit(m)
        temp_mark.remove(n)
        visited.add(n)
        sorted_files.append(n)
        
    for f in sorted(file_contents.keys()):
        if f not in visited:
            visit(f)

    # 5. Group by sections (Section mapping)
    section_map = {
        'CONSTANTS': [],
        'CONFIGURATION': [],
        'DATA MODELS': [],
        'MEMORY SYSTEM': [],
        'TOOLS': [],
        'AGENTS': [],
        'ORCHESTRATION': [],
        'API': [],
        'MAIN ENTRYPOINT': []
    }
    
    # Categorize
    for f in sorted_files:
        if f == 'main.py':
            section_map['MAIN ENTRYPOINT'].append(f)
        elif 'config' in f:
            section_map['CONFIGURATION'].append(f)
        elif 'models' in f or 'types' in f or 'registry' in f or 'capability' in f or 'risk_evaluator' in f or 'state_machine' in f or 'context' in f or 'profile' in f or 'contracts' in f:
            section_map['DATA MODELS'].append(f)
        elif 'memory/' in f:
            section_map['MEMORY SYSTEM'].append(f)
        elif 'tools/' in f or 'plugins/' in f or 'integrations/' in f or 'hardware/' in f or 'llm/' in f or 'desktop/' in f or 'voice/' in f:
            section_map['TOOLS'].append(f)
        elif 'agent/' in f or 'autonomy/' in f:
            section_map['AGENTS'].append(f)
        elif 'workflows/' in f or 'runtime/' in f or 'orchestration' in f or 'controller' in f or 'automation' in f or 'executor' in f or 'synthesis' in f or 'ops' in f or 'planner' in f:
            section_map['ORCHESTRATION'].append(f)
        elif 'api/' in f or 'dashboard/' in f:
            section_map['API'].append(f)
        else:
            # Safe defaults that don't depend on much
            if 'logger' in f or 'utils' in f or 'metrics' in f or 'paths' in f:
                section_map['CONSTANTS'].append(f)
            else:
                section_map['ORCHESTRATION'].append(f)

    # 6. Apply Text Replacements and build Monolith string
    out_lines = []
    out_lines.append('"""')
    out_lines.append('JARVIS MONOLITH')
    out_lines.append('=================')
    out_lines.append('LLM GUIDE: This file contains the entirety of the Jarvis project, merged from 150+ original modular files.')
    out_lines.append('How to navigate:')
    out_lines.append("1. FOLDER STRUCTURE: Before every file's content, there is a marker formatted as `# --- FILE: path/to/file.py ---`.")
    out_lines.append('   Use these markers to map the code back to its original modular location.')
    out_lines.append('2. RENAMED VARIABLES: To avoid namespace collisions during the merge, duplicate class/function names across files')
    out_lines.append('   were prefixed with their original paths (e.g., `ToolResult` from `core/tools/system_automation.py` became')
    out_lines.append('   `core_tools_system_automation_ToolResult`). You can safely trace these to understand execution.')
    out_lines.append('3. INTERNAL IMPORTS: Internal `from core.X import Y` statements were removed. Use direct references instead.')
    out_lines.append('4. WHEN EDITING: If you are suggesting fixes, please specify the ORIGINAL FILE PATH based on the markers,')
    out_lines.append('   rather than line numbers in this monolith, so the user can easily port your fix back to their repository.')
    out_lines.append('"""\n')
    out_lines.append('from __future__ import annotations\n')
    
    out_lines.append('############################################################')
    out_lines.append('# IMPORTS')
    out_lines.append('############################################################\n')
    for imp in sorted(global_imports):
        out_lines.append(imp)
    out_lines.append('\n')

    classes_merged = 0
    funcs_merged = 0

    for section_name, section_files in section_map.items():
        print(f"Processing section: {section_name} ({len(section_files)} files)")
        out_lines.append('############################################################')
        out_lines.append(f'# {section_name}')
        out_lines.append('############################################################\n')
        
        for rel in section_files:
            out_lines.append(f'\n# --- FILE: {rel} ---\n')
            content = file_contents[rel]
            tree = file_asts[rel]
            
            for node in tree.body:
                if isinstance(node, ast.ClassDef): classes_merged += 1
                elif isinstance(node, ast.FunctionDef): funcs_merged += 1
            
            lines = content.split('\n')
            
            for node in tree.body:
                # Remove if __name__ == '__main__'
                if rel != 'main.py' and isinstance(node, ast.If):
                    if isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__':
                        start = node.lineno - 1
                        end = getattr(node, 'end_lineno', len(lines))
                        for i in range(start, end):
                            lines[i] = f"# main block removed: {lines[i]}"
                            
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    is_internal = False
                    aliases_to_add = []
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            base_mod = alias.name.split('.')[0]
                            if base_mod in ['core', 'config', 'integrations', 'plugins', 'memory', 'workflows', 'runtime', 'dashboard', 'api', 'tools', 'DesignChange', 'LoopDesign', 'Cartography_Outputs']:
                                is_internal = True
                                if alias.asname:
                                    # In import a as b, b = a
                                    aliases_to_add.append(f"{alias.asname} = {alias.name.replace('.', '_')}") # Approximation
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module if node.module else ""
                        base_mod = module.split('.')[0]
                        if base_mod in ['core', 'config', 'integrations', 'plugins', 'memory', 'workflows', 'runtime', 'dashboard', 'api', 'tools', 'DesignChange', 'LoopDesign', 'Cartography_Outputs'] or node.level > 0:
                            is_internal = True
                            for alias in node.names:
                                if alias.asname:
                                    aliases_to_add.append(f"{alias.asname} = {alias.name}")
                        elif base_mod == '__future__':
                            is_internal = True
                    
                    if is_internal:
                        start = node.lineno - 1
                        end = getattr(node, 'end_lineno', node.lineno)
                        for i in range(start, end):
                            lines[i] = f"# internal import removed: {lines[i]}"
                        if aliases_to_add:
                            lines[end-1] += '\n' + '\n'.join(aliases_to_add)
            
            content = '\n'.join(lines)
            
            renames = file_renames.get(rel, {})
            for orig, new_name in renames.items():
                content = re.sub(rf'\b{orig}\b', new_name, content)
                
            out_lines.append(content)
            out_lines.append('\n')

    print("Writing files...")

    with open('jarvis_monolith.py', 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_lines))
        
    with open('monolith_dependency_graph.md', 'w', encoding='utf-8') as f:
        f.write("# Monolith Dependency Graph\n\n")
        f.write("```mermaid\ngraph TD;\n")
        for node, edges in deps.items():
            safe_node = node.replace('/', '_').replace('.py', '')
            for edge in edges:
                safe_edge = edge.replace('/', '_').replace('.py', '')
                f.write(f"    {safe_node} --> {safe_edge};\n")
        f.write("```\n")

    with open('monolith_migration_report.md', 'w', encoding='utf-8') as f:
        f.write("# Monolith Migration Report\n\n")
        f.write("## Duplicate Symbols Renamed\n")
        for (loc, name), new_name in rename_map.items():
            f.write(f"- `{name}` in `{loc}` renamed to `{new_name}`\n")
        f.write("\n## Metrics\n")
        f.write(f"- Files merged: {len(sorted_files)}\n")
        f.write(f"- Classes merged: {classes_merged}\n")
        f.write(f"- Functions merged: {funcs_merged}\n")
        f.write(f"- Final line count: {len(out_lines)}\n")

    print(f"Monolith built successfully! {classes_merged} classes, {funcs_merged} functions.")

if __name__ == '__main__':
    main()
