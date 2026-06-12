import os
import ast
from collections import defaultdict

def get_python_files(root_dir):
    py_files = []
    exclude_dirs = {
        '.git', '.mypy_cache', '.pytest_cache', '.ruff_cache', 'tests',
        'build', 'dist', 'htmlcov', 'venv', 'env', 'jarvis_env', 'min_venv', 'site-packages',
        'chroma_db', 'dashboard', 'bin', 'docs', 'data', 'logs'
    }
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for file in filenames:
            if file.endswith('.py') and not file.startswith('test_'):
                py_files.append(os.path.join(dirpath, file))
    return py_files

def main():
    root_dir = r"d:\AI\Jarvis"
    files = get_python_files(root_dir)
    
    defined_names = defaultdict(list)
    
    for filepath in files:
        rel = os.path.relpath(filepath, root_dir).replace('\\', '/')
        if rel in ['analyze_repo.py', 'analyze.py', 'api_analysis.py', 'automated_test.py', 'gen_docs.py', 'generate_cartography.py', 'qa_test.py', 'run_tests.py', 'scratch_script.py', 'test.py', 'check_duplicates.py', 'fix_cursor_leaks.py']:
            continue
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            tree = ast.parse(content, filename=filepath)
        except Exception as e:
            continue
            
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                defined_names[node.name].append(rel)
            elif isinstance(node, ast.FunctionDef):
                defined_names[node.name].append(rel)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        # only track uppercase variables as potential top-level constants
                        if target.id.isupper():
                            defined_names[target.id].append(rel)
                            
    duplicates = {name: files for name, files in defined_names.items() if len(files) > 1}
    for name, locations in duplicates.items():
        if name not in ['__init__', 'main', 'logger', 'get_logger', 'run']:
            print(f"Duplicate {name} in: {', '.join(locations)}")
            
    print(f"Total duplicates found: {len(duplicates)}")

if __name__ == '__main__':
    main()
