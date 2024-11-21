import aiosqlite
import asyncio

async def create_database(db_path):
    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                chat_name TEXT,
                role TEXT,
                content TEXT
            )
        """)
        await db.commit()

async def save_chat_history(chat_name, messages, db_path= "chat_history.db"):
    async with aiosqlite.connect(db_path) as db:
        await db.executemany(
            "INSERT INTO chat_history (chat_name, role, content) VALUES (?, ?, ?)",
            [(chat_name, message["role"], message["content"]) for message in messages]
        )
        await db.commit()

async def load_chat_history(chat_name, db_path="chat_history.db"):
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT role, content FROM chat_history WHERE chat_name = ? ORDER BY id",
            (chat_name,)
        ) as cursor:
            return [{"role": row[0], "content": row[1]} async for row in cursor]

async def get_chat_names(db_path="chat_history.db"):
    async with aiosqlite.connect(db_path) as db:
        async with db.execute("SELECT DISTINCT chat_name FROM chat_history") as cursor:
            return [row[0] async for row in cursor]