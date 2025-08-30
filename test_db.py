# test_db.py
import asyncio, sys
from sqlalchemy import text
from app.db.session import engine

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def main():
    async with engine.begin() as conn:
        result = await conn.execute(text("SELECT 1"))
        print(result.scalar())
    await engine.dispose()  # <- importante en scripts de prueba

if __name__ == "__main__":
    asyncio.run(main())
