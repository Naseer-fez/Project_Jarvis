import sqlite3
conn=sqlite3.connect('memory.db')
print('--- preferences ---')
for row in conn.execute('SELECT * FROM preferences'): print(row)
print('--- episodic_memory ---')
for row in conn.execute('SELECT * FROM episodic_memory'): print(row)
print('--- conversation_history ---')
for row in conn.execute('SELECT * FROM conversation_history'): print(row)
