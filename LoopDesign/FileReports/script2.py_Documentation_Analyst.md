# File Report: script2.py
**Role**: Documentation Analyst
**Target**: `d:\AI\Jarvis\memory\script2.py`

## Overview
A short, utilitarian Python script meant to dump all rows from the active tables in `memory.db`.

## Line-by-Line Analysis
- `1: import sqlite3`: Dependency on Python's built-in `sqlite3` library.
- `2: conn=sqlite3.connect('memory.db')`: Connects to local SQLite database.
- `3: print('--- preferences ---')`: Header for preferences table output.
- `4: for row in conn.execute('SELECT * FROM preferences'): print(row)`: Dumps all rows from `preferences`.
- `5: print('--- episodic_memory ---')`: Header for episodic_memory table output.
- `6: for row in conn.execute('SELECT * FROM episodic_memory'): print(row)`: Dumps all rows from `episodic_memory`.
- `7: print('--- conversation_history ---')`: Header for conversation_history table output.
- `8: for row in conn.execute('SELECT * FROM conversation_history'): print(row)`: Dumps all rows from `conversation_history`.

## Assumptions & Design Patterns
- **Selective Data Dump**: Only queries three specific tables (`preferences`, `episodic_memory`, `conversation_history`), confirming the assumption that these are the "active" or currently populated tables.

## Developer Notes
- No prompts found.
- Script is a transient developer tool for verifying data persistence.
