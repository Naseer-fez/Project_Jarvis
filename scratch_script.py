import json
import os

with open(r'd:\AI\Jarvis\core_files_utf8.json', 'r', encoding='utf-8-sig') as f:
    items = json.load(f)

folders = {}
root_files = []

for item in items:
    path = item['Path']
    typ = item['Type']
    if '\\' in path:
        folder, rest = path.split('\\', 1)
        if folder not in folders:
            folders[folder] = {'files': [], 'subfolders': []}
        if typ == 'File':
            folders[folder]['files'].append(rest)
        else:
            folders[folder]['subfolders'].append(rest)
    else:
        if typ == 'File':
            root_files.append(path)
        else:
            if path not in folders:
                folders[path] = {'files': [], 'subfolders': []}

def assign_roles(file_name, folder_name):
    roles = ["Runtime Investigator"]
    f = file_name.lower()
    
    if folder_name in ['config', 'ops', 'registry', 'security']:
        roles.append("Configuration Analyst")
    if folder_name in ['llm', 'planner'] or 'prompt' in f or 'synthesis' in f:
        roles.append("Prompt Recovery Specialist")
    if folder_name in ['memory', 'types', 'context'] or 'model' in f:
        roles.append("Data Model Analyst")
    if folder_name in ['tools', 'desktop', 'voice', 'hardware', 'automation', 'capability', 'controller'] or 'api' in f or 'client' in f:
        roles.append("API Analyst")
    if folder_name in ['runtime', 'executor', 'plugins', 'registry'] or 'init' in f or 'dependency' in f:
        roles.append("Dependency Analyst")

    return ", ".join(list(set(roles)))

with open(r'd:\AI\Jarvis\LoopDesign\FolderReports\core_manifest.md', 'w', encoding='utf-8') as out:
    out.write("# Core Directory Manifest\n\n")
    out.write("**Folder**: d:\\AI\\Jarvis\\core\n\n")
    out.write("**High-Level Purpose**: The core folder is the brain and primary engine of the Jarvis AI system. It encompasses the entirety of the execution pipeline, including LLM orchestration, semantic and episodic memory retrieval, external tool interfacing, desktop automation, autonomous goal management, hardware integrations, and system-level introspection. It orchestrates the flow of data from inputs (voice, text, proactive triggers) through the LLM planning layers, and dispatches actions to local or external environments.\n\n")
    
    out.write("## Root Level Files\n")
    for f in sorted(root_files):
        out.write(f"- {f} - Required Roles: {assign_roles(f, '')}\n")
    out.write("\n")
    
    for folder, contents in sorted(folders.items()):
        out.write(f"## Subdirectory: {folder}\n")
        out.write(f"- **Purpose**: Components and logic related to {folder} subsystem.\n")
        
        for f in sorted(contents['files']):
            out.write(f"- {folder}\\{f} - Required Roles: {assign_roles(f, folder)}\n")
        
        for sf in sorted(contents['subfolders']):
            out.write(f"- {folder}\\{sf} (Directory) - Required Roles: Dependency Analyst, Runtime Investigator\n")
        out.write("\n")

print('Done')
