from uuid import uuid4
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text as sqltext

async def register_file_source(session: AsyncSession, *, chat_id: str, name: str, file_url: str, file_name: str, file_size: int):
    source_id = str(uuid4())
    await session.execute(
        sqltext("""
            insert into public.knowledge_sources
              (id, chat_id, type, name, file_url, file_name, file_size, status)
            values
              (:id, :chat, 'file', :name, :url, :fname, :fsize, 'processing')
        """),
        {"id": source_id, "chat": chat_id, "name": name, "url": file_url, "fname": file_name, "fsize": file_size}
    )
    await session.commit()
    return source_id

async def register_text_source(session: AsyncSession, *, chat_id: str, name: str, content: str):
    source_id = str(uuid4())
    await session.execute(
        sqltext("""
            insert into public.knowledge_sources
              (id, chat_id, type, name, content, status)
            values
              (:id, :chat, 'text', :name, :content, 'processing')
        """),
        {"id": source_id, "chat": chat_id, "name": name, "content": content}
    )
    await session.commit()
    return source_id

async def init_file_source(session: AsyncSession, *, chat_id: str, name: str):
    source_id = str(uuid4())
    await session.execute(
        sqltext("""
            insert into public.knowledge_sources
              (id, chat_id, type, name, status)
            values
              (:id, :chat, 'file', :name, 'upload_pending')
        """),
        {"id": source_id, "chat": chat_id, "name": name}
    )
    await session.commit()
    return source_id

async def confirm_file_source_upload(session: AsyncSession, *, source_id: str, file_url: str, file_name: str, file_size: int):
    await session.execute(
        sqltext("""
            update public.knowledge_sources
            set file_url = :url, file_name = :fname, file_size = :fsize, status = 'processing'
            where id = :id
        """),
        {"id": source_id, "url": file_url, "fname": file_name, "fsize": file_size}
    )
    await session.commit()