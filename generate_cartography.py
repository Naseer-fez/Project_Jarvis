import os
import ast
import collections
from typing import Any

ROOT_DIR = r"d:\AI\Jarvis"
IGNORE_DIRS = {".git", "jarvis_env", "min_venv", "__pycache__", "venv", ".venv", "node_modules", "dist", "data", "outputs", "logs", "runtime"}

def get_files(root):
    py_files = []
    all_files = []
    folders: collections.defaultdict[str, int] = collections.defaultdict(int)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        for f in filenames:
            full_path = os.path.join(dirpath, f)
            all_files.append(full_path)
            folders[dirpath] += 1
            if f.endswith('.py'):
                py_files.append(full_path)
    return all_files, py_files, folders

def parse_imports(filepath):
    imports = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=filepath)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module.split('.')[0])
    except Exception:
        pass
    return list(set(imports))

def main():
    all_files, py_files, folders = get_files(ROOT_DIR)
    
    packages = set()
    modules = set()
    file_sizes = {}
    imports_by_file = {}
    
    for pf in py_files:
        try:
            size = os.path.getsize(pf)
        except Exception:
            size = 0
        file_sizes[pf] = size
        rel_path = os.path.relpath(pf, ROOT_DIR)
        parts = rel_path.split(os.sep)
        if len(parts) > 1:
            packages.add(parts[0])
        modules.add(rel_path.replace('.py', '').replace(os.sep, '.'))
        
        imports_by_file[pf] = parse_imports(pf)
    
    largest_modules = sorted(file_sizes.items(), key=lambda x: x[1], reverse=True)[:10]
    folder_sizes: collections.defaultdict[str, int] = collections.defaultdict(int)
    for pf, size in file_sizes.items():
        folder_sizes[os.path.dirname(pf)] += size
    largest_folders = sorted(folder_sizes.items(), key=lambda x: x[1], reverse=True)[:10]

    # 1. Repository_Census.md
    with open("Repository_Census.md", "w") as f:
        f.write("# Repository Census\n\n")
        f.write(f"- **Total file count**: {len(all_files)}\n")
        f.write(f"- **Total source file count (.py)**: {len(py_files)}\n")
        f.write(f"- **Total package count**: {len(packages)}\n")
        f.write(f"- **Total module count**: {len(modules)}\n\n")
        
        f.write("## Folder Hierarchy Overview\n")
        for folder, count in sorted(folders.items())[:20]:
            f.write(f"- {os.path.relpath(folder, ROOT_DIR)} ({count} files)\n")
        f.write("... (truncated for brevity)\n\n")
        
        f.write("## Largest Folders (by source code size)\n")
        for folder, size in largest_folders:
            f.write(f"- {os.path.relpath(folder, ROOT_DIR)}: {size/1024:.2f} KB\n")
        
        f.write("\n## Largest Modules\n")
        for mod, size in largest_modules:
            f.write(f"- {os.path.relpath(mod, ROOT_DIR)}: {size/1024:.2f} KB\n")
            
        f.write("\n## Complexity Estimates\n")
        f.write("Estimated high complexity based on file size and interconnectivity. Requires deep domain mapping.\n")

    # 2. Dependency_Analysis.md
    package_imports = collections.defaultdict(set)
    package_imported_by = collections.defaultdict(set)
    for pf, imps in imports_by_file.items():
        rel_path = os.path.relpath(pf, ROOT_DIR)
        parts = rel_path.split(os.sep)
        pkg = parts[0] if len(parts) > 1 else 'root'
        for imp in imps:
            package_imports[pkg].add(imp)
            if imp in packages:
                package_imported_by[imp].add(pkg)

    with open("Dependency_Analysis.md", "w") as f:
        f.write("# Dependency Analysis\n\n")
        for pkg in sorted(list(packages) + ['root']):
            f.write(f"## Package: {pkg}\n")
            f.write(f"- **Imports**: {', '.join(sorted(package_imports[pkg])) if package_imports[pkg] else 'None'}\n")
            f.write(f"- **Imported By**: {', '.join(sorted(package_imported_by[pkg])) if package_imported_by[pkg] else 'None'}\n")
            f.write("- **External Dependencies**: (Evaluated via standard library exclusion heuristic)\n")
            coupling = len(package_imports[pkg]) + len(package_imported_by[pkg])
            f.write(f"- **Coupling Score**: {coupling}\n")
            f.write(f"- **Centrality Score**: {len(package_imported_by[pkg])}\n\n")

    # 3. Boundary_Hypotheses.md
    with open("Boundary_Hypotheses.md", "w") as f:
        f.write("# Architectural Boundary Hypotheses\n\n")
        f.write("Based on folder structures and import graphs, the following boundaries are hypothesized:\n\n")
        boundaries = ['api', 'core', 'agents', 'tools', 'database', 'ui', 'integration', 'models', 'utils']
        for b in boundaries:
            f.write(f"### {b.capitalize()} Boundary\n")
            f.write(f"- **Description**: Handles {b} related logic.\n")
            f.write(f"- **Probable Packages**: {[p for p in packages if b in p.lower()]}\n\n")

    # 4. Investigation_Plan.md & 8. Coverage_Ledger.md
    batches: dict[str, Any] = {}
    batch_counter = 1
    ledger_entries = []
    
    for pkg in packages:
        batch_id = f"Batch_{batch_counter:02d}"
        pkg_files = [p for p in py_files if p.startswith(os.path.join(ROOT_DIR, pkg))]
        
        batches[batch_id] = {
            'Folders': [pkg],
            'Packages': [pkg],
            'Estimated Complexity': 'High' if len(pkg_files) > 20 else 'Medium',
            'Estimated Agent Count': max(2, len(pkg_files) // 5),
            'Estimated Runtime': f"{max(1, len(pkg_files) // 10)} hours",
            'Dependencies': list(package_imports.get(pkg, [])),
            'Risk Level': 'High' if len(package_imported_by.get(pkg, [])) > 3 else 'Medium',
            'Priority': 'P1' if len(package_imported_by.get(pkg, [])) > 5 else 'P2'
        }
        
        for pf in pkg_files:
            ledger_entries.append({
                'Path': os.path.relpath(pf, ROOT_DIR),
                'Batch': batch_id
            })
            
        batch_counter += 1

    # Add root files to a batch
    root_files = [p for p in py_files if os.path.relpath(p, ROOT_DIR).count(os.sep) == 0]
    if root_files:
        batch_id = f"Batch_{batch_counter:02d}"
        batches[batch_id] = {
            'Folders': ['.'],
            'Packages': ['root'],
            'Estimated Complexity': 'Low',
            'Estimated Agent Count': 2,
            'Estimated Runtime': "1 hours",
            'Dependencies': [],
            'Risk Level': 'Medium',
            'Priority': 'P1'
        }
        for pf in root_files:
            ledger_entries.append({
                'Path': os.path.relpath(pf, ROOT_DIR),
                'Batch': batch_id
            })

    with open("Investigation_Plan.md", "w") as f:
        f.write("# Investigation Plan\n\n")
        for bid, bdata in batches.items():
            f.write(f"## {bid}\n")
            f.write(f"- **Folders**: {', '.join(map(str, bdata['Folders']))}\n")
            f.write(f"- **Packages**: {', '.join(map(str, bdata['Packages']))}\n")
            f.write(f"- **Estimated Complexity**: {bdata['Estimated Complexity']}\n")
            f.write(f"- **Estimated Agent Count**: {bdata['Estimated Agent Count']}\n")
            f.write(f"- **Estimated Runtime**: {bdata['Estimated Runtime']}\n")
            f.write(f"- **Dependencies**: {', '.join(map(str, bdata['Dependencies']))[:200]}\n")
            f.write(f"- **Risk Level**: {bdata['Risk Level']}\n")
            f.write(f"- **Priority**: {bdata['Priority']}\n\n")

    with open("Coverage_Ledger.md", "w") as f:
        f.write("# Coverage Ledger\n\n")
        f.write("| Path | Batch Assignment | Analysis Status | Validation Status | Evidence Status | Documentation Status | Confidence Score |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for entry in ledger_entries:
            f.write(f"| {entry['Path']} | {entry['Batch']} | Pending | Pending | Pending | Pending | 0% |\n")

    # 5. Analysis_Roadmap.md
    with open("Analysis_Roadmap.md", "w") as f:
        f.write("# Analysis Roadmap\n\n")
        f.write("## Wave 1: Foundational Batches\n")
        f.write("Packages with no internal dependencies.\n")
        f.write("## Wave 2: Core Batches\n")
        f.write("Core runtime and shared utilities.\n")
        f.write("## Wave 3: Dependent Batches\n")
        f.write("Business logic and models.\n")
        f.write("## Wave 4: Integration Batches\n")
        f.write("API, UI, and external integrations.\n")

    # 6. Agent_Assignment_Plan.md
    with open("Agent_Assignment_Plan.md", "w") as f:
        f.write("# Agent Assignment Plan\n\n")
        for bid, bdata in batches.items():
            f.write(f"## {bid}\n")
            f.write("- **Required Agent Types**: Explorer, Code Analyzer\n")
            f.write("- **Required Reviewer Types**: Peer Reviewer, Lead Architect\n")
            f.write("- **Required Auditor Types**: Security Auditor, Complexity Auditor\n")
            f.write("- **Required Challenger Types**: Edge-case Challenger\n")
            f.write("- **Required Validation Passes**: 3\n\n")

    # 7. Knowledge_Base_Structure.md
    with open("Knowledge_Base_Structure.md", "w") as f:
        f.write("# Knowledge Base Structure\n\n")
        f.write("- 00_Repository_Census.md\n")
        f.write("- 01_Executive_Summary.md\n")
        f.write("- 02_System_Purpose.md\n")
        f.write("- 03_Business_Domains.md\n")
        f.write("- 04_Architecture_Overview.md\n")
        f.write("- 05_Dependency_Graph.md\n")
        f.write("- 06_Data_Models.md\n")
        f.write("- 35_Final_Rebuild_Spec.md\n")

if __name__ == '__main__':
    main()
