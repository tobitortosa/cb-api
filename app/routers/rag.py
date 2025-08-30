from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
import os

from app.db.session import get_session
from app.services.rag import (
    rag_answer,
    vector_search_text_and_tables,
    vector_search_img_desc,
)

router = APIRouter(prefix="/rag", tags=["rag"])

# =========================
# Config desde .env
# =========================
# top_k por defecto para /query
DEFAULT_RAG_TOP_K = int(os.getenv("RAG_TOP_K", "8"))
# top_k por defecto para /search (si no está, usa RAG_TOP_K)
DEFAULT_SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", str(DEFAULT_RAG_TOP_K)))
# Máximo de resultados de imágenes que se permiten en /search
SEARCH_IMG_K = int(os.getenv("SEARCH_IMG_K", "4"))

# Defaults de generación (modelo y temperatura)
DEFAULT_RAG_MODEL = os.getenv("RAG_MODEL", "gpt-4o-mini")
DEFAULT_RAG_TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", "0.2"))

# (opcional) límites sanos
TOP_K_MIN = int(os.getenv("TOP_K_MIN", "1"))
TOP_K_MAX = int(os.getenv("TOP_K_MAX", "20"))
TEMP_MIN = float(os.getenv("TEMP_MIN", "0.0"))
TEMP_MAX = float(os.getenv("TEMP_MAX", "2.0"))

def _clamp_k(k: int, mn: int = TOP_K_MIN, mx: int = TOP_K_MAX) -> int:
    try:
        return max(mn, min(mx, int(k)))
    except Exception:
        return mn

def _clamp_temp(t: float, mn: float = TEMP_MIN, mx: float = TEMP_MAX) -> float:
    try:
        return max(mn, min(mx, float(t)))
    except Exception:
        return mn

# =========================
# Modelos de request/response
# =========================

class ChatMessage(BaseModel):
    role: str = Field(pattern=r"^(system|user|assistant)$")
    content: str

class QueryBody(BaseModel):
    messages: List[ChatMessage]
    # ranking
    top_k: Optional[int] = None
    # generación del LLM (opcionales)
    model: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    # persona/template elegido por el usuario (opcional)
    system_prompt: Optional[str] = Field(default=None, max_length=20000)
    # cómo combinar el template con el core: merge | replace | core_only
    system_prompt_mode: Optional[str] = Field(
        default="merge",
        pattern=r"^(merge|replace|core_only)$"
    )
    # permitir override de max_tokens si querés
    max_tokens: Optional[int] = Field(default=None, ge=1)

@router.post("/chats/{chat_id}/query")
async def rag_query(
    chat_id: UUID,
    body: QueryBody,
    session: AsyncSession = Depends(get_session),
):
    """
    Consulta RAG conversacional:
      - Recupera top_k chunks ('text' | 'table' e 'img_text' si corresponde).
      - Llama al modelo con grounding estricto.
      - Permite configurar model, temperature, system_prompt (persona) y system_prompt_mode.
    """
    try:
        # Efectivos con defaults y clamps
        effective_top_k = _clamp_k(body.top_k if body.top_k is not None else DEFAULT_RAG_TOP_K)
        effective_model = (body.model or DEFAULT_RAG_MODEL).strip()
        effective_temp = _clamp_temp(body.temperature if body.temperature is not None else DEFAULT_RAG_TEMPERATURE)

        # Importante: si no se envía system_prompt, dejamos None para no duplicar el core
        persona_prompt = body.system_prompt if (body.system_prompt and body.system_prompt.strip()) else None
        prompt_mode = (body.system_prompt_mode or "merge").strip()

        res = await rag_answer(
            session=session,
            chat_id=chat_id,
            messages=[m.model_dump() for m in body.messages],
            top_k=effective_top_k,
            model=effective_model,
            temperature=effective_temp,
            system_prompt=persona_prompt,     # template/persona opcional
            prompt_mode=prompt_mode,          # merge|replace|core_only
            max_tokens=body.max_tokens,       # opcional
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:400])

# ---- Resultados de /search para UI ----

class SearchResult(BaseModel):
    id: str
    kind: str                           # 'text' | 'table' | 'img_text'
    page: Optional[int]
    score: float
    preview: str
    html: Optional[str] = None          # solo para 'table'
    # Extras para 'img_text'
    image_uid: Optional[str] = None
    bbox: Optional[List[float]] = None
    bbox_norm: Optional[List[float]] = None
    order_index: Optional[int] = None
    tags: Optional[List[str]] = None
    objects: Optional[List[str]] = None
    colors: Optional[dict] = None
    # Extras opcionales para 'table' por si querés overlay en UI
    table_bbox: Optional[List[float]] = None
    table_bbox_norm: Optional[List[float]] = None
    table_order_index: Optional[int] = None

@router.get("/chats/{chat_id}/search")
async def rag_search(
    chat_id: UUID,
    q: str,
    top_k: Optional[int] = None,
    session: AsyncSession = Depends(get_session),
):
    """
    Devuelve hits para UI de “fuentes”.
    Limita la cantidad de 'img_text' a SEARCH_IMG_K para no saturar la lista.
    """
    try:
        effective_top_k = _clamp_k(top_k if top_k is not None else DEFAULT_SEARCH_TOP_K)

        # 1) Buscar texto+tabla
        hits_tt = await vector_search_text_and_tables(session, chat_id, q, top_k=effective_top_k)

        # 2) Buscar imágenes (limitamos la mezcla)
        img_k = min(SEARCH_IMG_K, max(1, effective_top_k))
        hits_img = await vector_search_img_desc(session, chat_id, q, top_k=img_k)

        # 3) Merge + dedupe por id
        merged = []
        seen = set()
        for h in hits_tt + hits_img:
            hid = h["id"]
            if hid in seen:
                continue
            seen.add(hid)
            merged.append(h)

        # 4) Ordenar por score desc
        merged.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)

        # 5) Recortar manteniendo un límite de imágenes en la lista final
        out: List[SearchResult] = []
        img_used = 0
        for h in merged:
            kind = h["kind"]
            meta = h.get("meta") or {}

            if kind == "img_text" and img_used >= SEARCH_IMG_K:
                continue

            if kind == "table":
                plain = (meta.get("plain") or "").strip()
                html  = h.get("content") or ""
                preview = plain or html
                if len(preview) > 600:
                    preview = preview[:600] + "…"

                out.append(SearchResult(
                    id=h["id"],
                    kind=kind,
                    page=h.get("page"),
                    score=h["score"],
                    preview=preview,
                    html=html,
                    table_bbox=meta.get("bbox"),
                    table_bbox_norm=meta.get("bbox_norm"),
                    table_order_index=meta.get("order_index"),
                ))

            elif kind == "img_text":
                preview = (h.get("content") or "")
                if len(preview) > 600:
                    preview = preview[:600] + "…"

                out.append(SearchResult(
                    id=h["id"],
                    kind=kind,
                    page=h.get("page"),
                    score=h["score"],
                    preview=preview,
                    image_uid=meta.get("image_uid"),
                    bbox=meta.get("bbox"),
                    bbox_norm=meta.get("bbox_norm"),
                    order_index=meta.get("order_index"),
                    tags=meta.get("tags"),
                    objects=meta.get("objects"),
                    colors=meta.get("colors"),
                ))
                img_used += 1

            else:
                preview = (h.get("content") or "")
                if len(preview) > 600:
                    preview = preview[:600] + "…"

                out.append(SearchResult(
                    id=h["id"],
                    kind=kind,
                    page=h.get("page"),
                    score=h["score"],
                    preview=preview,
                ))

            if len(out) >= effective_top_k:
                break

        return out

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:400])