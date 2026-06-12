import sqlite3
conn=sqlite3.connect('memory.db')
tables=['preferences','episodic_memory','conversation_history','episodes','facts','conversations','actions']
for t in tables:
    try: print(f'{t}: {conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] }')
    except: pass
