import asyncio
import aiosqlite

async def main():
    async with aiosqlite.connect(":memory:") as conn:
        async with conn.execute("CREATE TABLE test (id INT);"):
            pass
        
        async with conn.executemany("INSERT INTO test VALUES (?)", [(1,), (2,)]):
            pass
            
        async with conn.executescript("SELECT 1; SELECT 2;"):
            pass
            
        print("Success")

asyncio.run(main())
