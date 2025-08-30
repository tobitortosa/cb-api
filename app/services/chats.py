from uuid import uuid4
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text as sqltext

async def create_chat(session: AsyncSession, *, user_id: str, name: str):
    chat_id = str(uuid4())
    await session.execute(
        sqltext("""
            insert into public.chats (id, user_id, name, status)
            values (:id, :uid, :name, 'draft')
        """),
        {"id": chat_id, "uid": user_id, "name": name}
    )
    await session.commit()
    return chat_id