# app/routers/chats.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text as sqltext
from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID
from app.db.session import get_session
from app.middlewares.auth import current_user
from app.schemas.chat import ChatCreate, ChatOut
from app.schemas.source import (
    SourceCreate, SourceCreateText, SourceCreateFile, SourceOut,
    SourceInitIn, SourceConfirmUploadIn
)
from app.services.chats import create_chat
from app.services.sources import (
    init_file_source, confirm_file_source_upload, register_text_source
)
from app.services.ingest import ingest_text_source, ingest_file_source

router = APIRouter(prefix="/chats", tags=["chats"])

# === PYDANTIC MODELS FOR TEXT SOURCES ===
class TextSourceBody(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=200_000)

class TextSourceUpdateBody(BaseModel):
    title: Optional[str] = Field(default=None, max_length=200)
    content: Optional[str] = Field(default=None, max_length=200_000)

@router.post("", response_model=ChatOut)
async def create_chat_route(
    payload: ChatCreate,
    user = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    chat_id = await create_chat(session, user_id=user["sub"], name=payload.name)
    return {"chat_id": chat_id}

# === NUEVO: init de fuente de archivo ===
@router.post("/{chat_id}/sources/init", response_model=SourceOut)
async def init_source_route(
    chat_id: str,
    payload: SourceInitIn | None = None,   # opcional: { "name": "Manual.pdf" }
    user = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    print("datos")
    print(chat_id)
    print(payload)
    print(user)
    print(session)
  
    # Validar que el chat sea del usuario
    r = await session.execute(
        sqltext("select 1 from public.chats where id=:id and user_id=:uid"),
        {"id": chat_id, "uid": user["sub"]}
    )
    if not r.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Chat no encontrado")

    name = (payload.name if payload else None) or "File source"
    source_id = await init_file_source(session, chat_id=chat_id, name=name)
    return {"source_id": source_id, "status": "upload_pending"}

# === NUEVO: confirmar upload y disparar ingesta ===
@router.patch("/{chat_id}/sources/{source_id}", response_model=SourceOut)
async def confirm_source_upload_route(
    chat_id: str,
    source_id: str,
    payload: SourceConfirmUploadIn,
    bg: BackgroundTasks,
    user = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    # Validar pertenencia
    r = await session.execute(sqltext("""
        select 1
        from public.knowledge_sources s
        join public.chats c on c.id = s.chat_id
        where s.id = :sid and s.chat_id = :cid and c.user_id = :uid
    """), {"sid": source_id, "cid": chat_id, "uid": user["sub"]})
    if not r.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Fuente no encontrada")

    await confirm_file_source_upload(
        session,
        source_id=source_id,
        file_url=payload.file_url,
        file_name=payload.file_name,
        file_size=payload.file_size
    )

    # Disparar ingesta de archivo (PDF → extracts → chunks)
    bg.add_task(ingest_file_source, session, chat_id=chat_id, source_id=source_id)

    return {"source_id": source_id, "status": "processing"}

# === EXISTENTE: crear fuente (quedátelo para TEXTO) ===
@router.post("/{chat_id}/sources", response_model=SourceOut)
async def create_source_route(
    chat_id: str,
    payload: SourceCreate,
    bg: BackgroundTasks,
    user = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    # Validar chat
    r = await session.execute(
        sqltext("select 1 from public.chats where id=:id and user_id=:uid"),
        {"id": chat_id, "uid": user["sub"]}
    )
    if not r.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Chat no encontrado")

    if isinstance(payload, SourceCreateText):
        source_id = await register_text_source(
            session, chat_id=chat_id, name=payload.name, content=payload.content
        )
        bg.add_task(
            ingest_text_source, session,
            chat_id=chat_id, source_id=source_id,
            name=payload.name, content=payload.content
        )
        return {"source_id": source_id, "status": "processing"}

    # (opcional) si querés, devolvé 400 para file acá para forzar el flujo init/patch
    if isinstance(payload, SourceCreateFile):
        raise HTTPException(status_code=400, detail="Usá /sources/init y luego PATCH para archivos")

    raise HTTPException(status_code=400, detail="Tipo de fuente inválido")

# === NUEVO: crear fuente de texto ===
@router.post("/{chat_id}/sources:text")
async def create_text_source(
    chat_id: UUID, 
    body: TextSourceBody, 
    session: AsyncSession = Depends(get_session)
):
    # 1) creo la fila en knowledge_sources
    row = (await session.execute(sqltext("""
        insert into public.knowledge_sources
          (chat_id, type, name, content, status, character_count)
        values
          (:chat, 'text', :name, :content, 'processing', length(:content))
        returning id
    """), {"chat": str(chat_id), "name": body.title.strip(), "content": body.content})).first()
    source_id = row[0]
    await session.commit()

    # 2) indexo inmediatamente
    result = await ingest_text_source(
        session,
        chat_id=str(chat_id),
        source_id=str(source_id),
        name=body.title,
        content=body.content,
    )
    return {"ok": True, "source_id": str(source_id), **result}

# === NUEVO: editar fuente de texto ===
@router.put("/{chat_id}/sources/{source_id}:text")
async def update_text_source(
    chat_id: UUID, 
    source_id: UUID, 
    body: TextSourceUpdateBody, 
    session: AsyncSession = Depends(get_session)
):
    from app.services.ingest import ingest_text_source_from_db
    
    # actualizo name/content
    await session.execute(sqltext("""
        update public.knowledge_sources
           set name = coalesce(:name, name),
               content = coalesce(:content, content),
               status = 'processing',
               character_count = case when :content is null then character_count else length(:content) end,
               updated_at = now()
         where id = :sid and chat_id = :cid and type = 'text'
    """), {"name": body.title, "content": body.content, "sid": str(source_id), "cid": str(chat_id)})
    await session.commit()
    
    # reindexo leyendo desde DB
    result = await ingest_text_source_from_db(session=session, chat_id=chat_id, source_id=source_id)
    return {"ok": True, "source_id": str(source_id), **result}

@router.get("/{chat_id}/sources")
async def list_sources_route(
    chat_id: str,
    user = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    r = await session.execute(sqltext("""
        select id, type, name, status, file_name, file_size, character_count, error_message, created_at
        from public.knowledge_sources
        where chat_id = :id
        order by created_at desc
    """), {"id": chat_id})
    return [
        {
            "id": str(row.id),
            "type": row.type,
            "name": row.name,
            "status": row.status,
            "file_name": row.file_name,
            "file_size": row.file_size,
            "character_count": row.character_count,
            "error_message": row.error_message,
            "created_at": row.created_at,
        }
        for row in r
    ]