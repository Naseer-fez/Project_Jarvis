import os
import ast
import json

def get_python_files(root_dir):
    py_files = []
    # Directories to exclude
    exclude_dirs = {
        '.git', '.mypy_cache', '.pytest_cache', '.ruff_cache', 'tests',
        'build', 'dist', 'htmlcov', 'venv', 'env', 'jarvis_env', 'min_venv', 'site-packages',
        'chroma_db', 'dashboard', 'bin', 'docs', 'data', 'logs'
    }
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # modify dirnames in-place to prune excluded directories
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for file in filenames:
            if file.endswith('.py') and not file.startswith('test_'):
                py_files.append(os.path.join(dirpath, file))
    return py_files

def analyze_imports(filepath, project_root):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        tree = ast.parse(content, filename=filepath)
    except Exception as e:
        return {'error': str(e)}

    imports = []
    from_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module if node.module else ''
            names = [alias.name for alias in node.names]
            from_imports.append({'module': module, 'level': node.level, 'names': names})
            
    # Calculate relative path
    rel_path = os.path.relpath(filepath, project_root).replace('\\', '/')
    return {
        'rel_path': rel_path,
        'imports': imports,
        'from_imports': from_imports
    }

def main():
    root_dir = r"d:\AI\Jarvis"
    files = get_python_files(root_dir)
    analysis = {}
    for f in files:
        # Ignore scratch or build scripts in root if necessary
        rel = os.path.relpath(f, root_dir).replace('\\', '/')
        if rel in ['analyze_repo.py', 'analyze.py', 'api_analysis.py', 'automated_test.py', 'gen_docs.py', 'generate_cartography.py', 'qa_test.py', 'run_tests.py', 'scratch_script.py', 'test.py']:
            continue
        analysis[rel] = analyze_imports(f, root_dir)

    with open('repo_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2)
        
    print(f"Analyzed {len(analysis)} files. Saved to repo_analysis.json")

if __name__ == '__main__':
    main()
