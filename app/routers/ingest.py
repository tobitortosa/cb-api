from fastapi import APIRouter, Depends, HTTPException, status, Query
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text as sqltext
from app.db.session import get_session
from app.services.ingest import ingest_source_service, ingest_text_source_from_db

router = APIRouter(prefix="/chats", tags=["ingest"])

@router.post("/{chat_id}/sources/{source_id}/ingest")
async def ingest_source(
    chat_id: UUID, 
    source_id: UUID, 
    force_reprocess: bool = Query(False, description="Force reprocessing even if already processed"),
    session: AsyncSession = Depends(get_session)
):
    try:
        # 1) Verificar que la fuente existe y obtener información
        source_info = (await session.execute(sqltext("""
            select type, status, character_count, file_url, file_name, file_size, error_message from public.knowledge_sources
             where id=:sid and chat_id=:cid
            limit 1
        """), {"sid": str(source_id), "cid": str(chat_id)})).mappings().first()

        if not source_info:
            raise HTTPException(status_code=404, detail="source not found for this chat")

        kind = source_info["type"]
        current_status = source_info["status"]
        current_char_count = source_info["character_count"] or 0
        file_url = source_info.get("file_url")
        file_name = source_info.get("file_name")
        file_size = source_info.get("file_size")
        error_message = source_info.get("error_message")

        # Para archivos, verificar que tengan file_url
        if kind == "file" and not file_url:
            raise HTTPException(status_code=400, detail="File source has no file_url. Upload may not be complete.")
        
        # Si el estado es 'failed', mostrar el error
        if current_status == "failed":
            error_detail = f"Source processing failed: {error_message}" if error_message else "Source processing failed with unknown error"
            raise HTTPException(status_code=422, detail=error_detail)

        # 2) Verificar si ya existen chunks para esta fuente
        existing_chunks = (await session.execute(sqltext("""
            select count(*) as chunk_count,
                   count(case when kind = 'text' then 1 end) as text_chunks,
                   count(case when kind = 'table' then 1 end) as table_chunks,
                   count(case when kind = 'img_text' then 1 end) as image_chunks
            from public.chunks
            where source_id = :sid
        """), {"sid": str(source_id)})).mappings().first()

        total_chunks = existing_chunks["chunk_count"] or 0

        # 3) Si ya existen chunks y el estado es 'active', retornar información existente (a menos que se fuerce el reprocesamiento)
        if total_chunks > 0 and current_status == "active" and not force_reprocess:
            print(f"[INGEST] Fuente {source_id} ya procesada. Chunks existentes: {total_chunks}")
            
            # Obtener estadísticas adicionales para archivos
            if kind == "file":
                # Obtener información del manifest en content
                content_info = (await session.execute(sqltext("""
                    select content from public.knowledge_sources
                    where id = :sid
                """), {"sid": str(source_id)})).scalar_one_or_none()
                
                if content_info:
                    try:
                        import json
                        manifest = json.loads(content_info)
                        stats = manifest.get("stats", {})
                        return {
                            "ok": True,
                            "already_processed": True,
                            "pages": stats.get("pages", 0),
                            "tables": stats.get("tables", 0),
                            "images": stats.get("images", 0),
                            "images_indexed": stats.get("images_indexed", 0),
                            "characters": current_char_count,
                            "chunks": {
                                "total": total_chunks,
                                "text": existing_chunks["text_chunks"],
                                "tables": existing_chunks["table_chunks"],
                                "images": existing_chunks["image_chunks"]
                            }
                        }
                    except (json.JSONDecodeError, AttributeError):
                        pass
            
            # Para fuentes de texto o si no hay manifest
            return {
                "ok": True,
                "already_processed": True,
                "characters": current_char_count,
                "chunks": {
                    "total": total_chunks,
                    "text": existing_chunks["text_chunks"],
                    "tables": existing_chunks["table_chunks"],
                    "images": existing_chunks["image_chunks"]
                }
            }

        # 4) Si no existen chunks, el estado no es 'active', o se fuerza reprocesamiento, procesar normalmente
        if force_reprocess and total_chunks > 0:
            print(f"[INGEST] Forzando reprocesamiento de fuente {source_id}")
        
        print(f"[INGEST] Procesando fuente {source_id}. Chunks existentes: {total_chunks}, Estado: {current_status}")

        if kind == "text":
            result = await ingest_text_source_from_db(session=session, chat_id=chat_id, source_id=source_id)
        elif kind == "file":
            result = await ingest_source_service(session=session, chat_id=chat_id, source_id=source_id)
        else:
            raise HTTPException(status_code=400, detail=f"unsupported source type: {kind}")

        return {"ok": True, "already_processed": False, **result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))