import asyncio
import os

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

load_dotenv()
engine = create_async_engine(os.environ["DATABASE_URL"])


async def run():
    async with engine.connect() as c:
        r = await c.execute(
            text(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name"
            )
        )
        tables = [row[0] for row in r.fetchall()]
        print("TABLES:", tables)
        for t in tables:
            r2 = await c.execute(
                text(
                    f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name='{t}' ORDER BY ordinal_position"
                )
            )
            print(f"\n--- {t} ---")
            for row in r2.fetchall():
                print(f"  {row[0]}: {row[1]}")
    await engine.dispose()


asyncio.run(run())
