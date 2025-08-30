import os, re, json
from typing import Any, Dict, List, Optional
from uuid import UUID
from sqlalchemy import text as sqltext
from sqlalchemy.ext.asyncio import AsyncSession
from openai import OpenAI

from app.services.ingest import embed_text  # mismo modelo de embeddings

# =========================
# Config
# =========================
OPENAI_MODEL_FALLBACK = os.getenv("ANSWER_MODEL", "gpt-4o-mini")
ANSWER_TEXT_MODEL     = os.getenv("ANSWER_TEXT_MODEL", "gpt-4o-mini")
DEFAULT_RAG_TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", "0.2"))

# Presupuesto de contexto y mezcla
MAX_SOURCE_CHARS    = int(os.getenv("MAX_SOURCE_CHARS", "12000"))
RAG_TOP_K           = int(os.getenv("RAG_TOP_K", "8"))          # top_k base (texto+tabla)
RAG_IMG_MIX_K       = int(os.getenv("RAG_IMG_MIX_K", "6"))      # cuántas candidatas de imágenes buscar
RAG_IMG_MIX_LIMIT   = int(os.getenv("RAG_IMG_MIX_LIMIT", "3"))  # tope de imágenes a incluir en contexto
IMG_MIN_SCORE       = float(os.getenv("IMG_MIN_SCORE", "0.35")) # umbral duro mínimo para imágenes
IMG_SCORE_MARGIN    = float(os.getenv("IMG_SCORE_MARGIN", "0.03")) # margen relativo vs. peor texto del top_k

# Historial
HISTORY_MAX_TURNS      = int(os.getenv("HISTORY_MAX_TURNS", "8"))
HISTORY_MAX_CHARS      = int(os.getenv("HISTORY_MAX_CHARS", "6000"))
RETRIEVAL_HINT_CHARS   = int(os.getenv("RETRIEVAL_HINT_CHARS", "900"))

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# Helpers
# =========================
def _trim(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else (s[:n] + "…")

def _normalize(s: str) -> str:
    return (s or "").strip().lower()

async def get_chat_row(session: AsyncSession, chat_id: UUID) -> Dict[str, Any]:
    row = (await session.execute(sqltext("""
        SELECT id, name, system_prompt, show_sources, model, temperature, max_tokens
        FROM public.chats
        WHERE id = :cid
        LIMIT 1
    """), {"cid": str(chat_id)})).mappings().first()
    if not row:
        raise ValueError("Chat not found")
    return dict(row)

def _slice_recent_dialog(messages: List[Dict[str, str]], max_turns: int, max_chars: int) -> List[Dict[str, str]]:
    dialog = [m for m in messages if m.get("role") in ("user", "assistant")]
    dialog = dialog[-(max_turns*2):] if max_turns > 0 else dialog
    total = 0
    out_rev: List[Dict[str, str]] = []
    for m in reversed(dialog):
        c = len((m.get("content") or ""))
        if total + c > max_chars:
            break
        out_rev.append(m)
        total += c
    return list(reversed(out_rev))

def _history_hint(messages: List[Dict[str, str]], exclude_last_user: bool = True, max_chars: int = 900) -> str:
    if not messages:
        return ""
    msgs = [m for m in messages if m.get("role") in ("user","assistant")]
    if exclude_last_user and msgs and msgs[-1].get("role") == "user":
        msgs = msgs[:-1]
    text = ""
    for m in msgs[-8:]:
        prefix = "U:" if m["role"] == "user" else "A:"
        chunk = f"{prefix} {m.get('content','').strip()}\n"
        if len(text) + len(chunk) > max_chars:
            break
        text += chunk
    return text.strip()

# =========================
# Vector search
# =========================
async def vector_search_text_and_tables(
    session: AsyncSession,
    chat_id: UUID,
    query: str,
    top_k: int,
    min_score: float = 0.0,
) -> List[Dict[str, Any]]:
    q_emb = await embed_text(query)
    qemb = "[" + ",".join(f"{x:.6f}" for x in q_emb) + "]"
    sql = sqltext("""
        SELECT id, kind, content, page, meta,
               1 - (embedding <=> CAST(:qemb AS text)::vector) AS score
        FROM public.chunks
        WHERE chat_id = :chat
          AND kind IN ('text','table')
          AND embedding IS NOT NULL
        ORDER BY embedding <=> CAST(:qemb AS text)::vector ASC
        LIMIT :lim
    """)
    rows = (await session.execute(sql, {"qemb": qemb, "chat": str(chat_id), "lim": int(top_k)})).mappings().all()

    out = []
    for r in rows:
        sc = float(r["score"] or 0.0)
        if sc >= min_score:
            meta = r["meta"] if isinstance(r["meta"], dict) else json.loads(r["meta"] or "{}")
            out.append({
                "id": str(r["id"]),
                "kind": r["kind"],        # 'text' | 'table'
                "content": r["content"],
                "page": r["page"],
                "meta": meta,
                "score": sc,
            })
    return out

async def vector_search_img_desc(
    session: AsyncSession,
    chat_id: UUID,
    query: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    q_emb = await embed_text(query)
    qemb = "[" + ",".join(f"{x:.6f}" for x in q_emb) + "]"
    sql = sqltext("""
        SELECT id, kind, content, page, meta,
               1 - (embedding <=> CAST(:qemb AS text)::vector) AS score
        FROM public.chunks
        WHERE chat_id = :chat
          AND kind = 'img_text'
          AND embedding IS NOT NULL
        ORDER BY embedding <=> CAST(:qemb AS text)::vector ASC
        LIMIT :lim
    """)
    rows = (await session.execute(sql, {"qemb": qemb, "chat": str(chat_id), "lim": int(top_k)})).mappings().all()
    out = []
    for r in rows:
        meta = r["meta"] if isinstance(r["meta"], dict) else json.loads(r["meta"] or "{}")
        out.append({
            "id": str(r["id"]),
            "kind": "img_text",
            "content": r["content"],
            "page": r["page"],
            "meta": meta,
            "score": float(r["score"] or 0.0),
        })
    return out

# =========================
# System prompt (core + persona)
# =========================

# Reglas core no negociables del sistema (RAG + formato)
CORE_SYSTEM_RULES = (
    "Eres un asistente conversacional para un sistema RAG.\n"
    "Dispones de dos fuentes: (a) el historial de conversación y (b) un <document_context> con fragmentos del/los documento(s).\n"
    "Usa el historial para la charla general. Usa <document_context> SOLO para responder sobre el/los documento(s).\n\n"
    "Reglas de grounding estrictas:\n"
    "1) No inventes datos: si un dato (nombre, fecha, valor) no aparece explícito en <document_context>, indica que no puedes confirmarlo.\n"
    "2) No aceptes correcciones del usuario si no están también en <document_context>.\n"
    "3) Si hay ambigüedad o evidencia insuficiente, pide una aclaración breve.\n"
    "4) No copies etiquetas internas ([TEXT], [IMG], [TABLE]) en la respuesta final.\n"
    "5) Responde en HTML simple usando SOLO: <p>, <ul>, <ol>, <li>, <b>, <i>, <code>, <pre>, <table>, <thead>, <tbody>, <tr>, <th>, <td>, <br>.\n"
    "6) Para preguntas directas de campo (p.ej., “fecha de emisión”), prioriza una respuesta breve con el valor.\n"
)

def _adapt_persona_template_for_rag(text: str) -> str:
    """
    Ajustes suaves para que los templates encajen con RAG.
    - 'training data' -> '<document_context> y el historial de la conversación'
    - Añade recordatorio de respetar reglas core.
    """
    if not text:
        return text
    t = text
    t = re.sub(r"\btraining data\b", "<document_context> y el historial de la conversación", t, flags=re.I)
    t = re.sub(r"\bprovided data\b", "<document_context>", t, flags=re.I)
    append_note = (
        "\n\n[Nota] Debes respetar las reglas del sistema y el uso de <document_context>. "
        "En caso de conflicto, obedece primero las reglas del sistema."
    )
    if "Debes respetar las reglas del sistema" not in t:
        t += append_note
    return t.strip()

def _compose_system_messages(
    chat_row: Dict[str, Any],
    *,
    persona_prompt: Optional[str],
    prompt_mode: str = "merge",  # 'merge' | 'replace' | 'core_only'
) -> List[str]:
    """
    Devuelve una lista de contenidos para mensajes `system`, en orden.
    - merge (default): [CORE_SYSTEM_RULES] + [chat_row.system_prompt?] + [persona?]
    - replace: [persona] (reemplaza TODO; NO recomendado)
    - core_only: [CORE_SYSTEM_RULES] (+ chat_row.system_prompt opcional)
    """
    sys_msgs: List[str] = []
    mode = (prompt_mode or "merge").lower().strip()
    if mode not in {"merge", "replace", "core_only"}:
        mode = "merge"

    if mode == "replace":
        if persona_prompt and persona_prompt.strip():
            sys_msgs.append(persona_prompt.strip())
        else:
            sys_msgs.append(CORE_SYSTEM_RULES)
        return sys_msgs

    # core_only o merge
    sys_msgs.append(CORE_SYSTEM_RULES)

    chat_sys = (chat_row.get("system_prompt") or "").strip()
    if chat_sys:
        sys_msgs.append(chat_sys)

    if mode == "core_only":
        return sys_msgs

    # merge: añadimos persona adaptada a RAG
    if persona_prompt and persona_prompt.strip():
        sys_msgs.append(_adapt_persona_template_for_rag(persona_prompt.strip()))
    return sys_msgs

# =========================
# Construcción de contexto RAG
# =========================
def _block_for_hit(h: Dict[str, Any]) -> str:
    kind = h.get("kind")
    meta = h.get("meta") or {}
    score = h.get("score", 0.0)
    page = h.get("page", "?")

    if kind == "text" and meta.get("kind") == "doc_summary":
        return f"[DOC SUMMARY score={score:.3f}]\n{_trim(h.get('content') or '', 2000)}"

    if kind == "table":
        plain = (meta.get("plain") or "").strip()
        text = plain if plain else _trim(h.get("content") or "", 1500)
        return f"[TABLE @p{page} score={score:.3f}]\n{text}"

    if kind == "img_text":
        return f"[IMG @p{page} score={score:.3f}]\n{_trim(h.get('content') or '', 1800)}"

    return f"[TEXT @p{page} score={score:.3f}]\n{_trim(h.get('content') or '', 1800)}"

def _make_context_string(hits: List[Dict[str, Any]], max_chars: int) -> str:
    used = 0
    parts: List[str] = []
    for h in hits:
        block = _block_for_hit(h)
        if not block.strip():
            continue
        if used + len(block) > max_chars:
            parts.append(_trim(block, max_chars - used))
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)

# =========================
# RAG principal (history-aware, sin heurísticas de intención)
# =========================
async def rag_answer(
    session: AsyncSession,
    *,
    chat_id: UUID,
    messages: List[Dict[str, str]],
    top_k: int = RAG_TOP_K,
    # overrides opcionales desde el endpoint
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    system_prompt: Optional[str] = None,   # persona/template elegido por el usuario
    max_tokens: Optional[int] = None,
    prompt_mode: str = "merge",            # 'merge' | 'replace' | 'core_only'
) -> Dict[str, Any]:
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("El último mensaje debe ser del usuario")

    # Historial recortado (para charla general)
    history_window = _slice_recent_dialog(messages, HISTORY_MAX_TURNS, HISTORY_MAX_CHARS)

    # Query de recuperación = último user + hint compacto del historial
    last_user_msg = messages[-1]["content"]
    hint = _history_hint(messages, exclude_last_user=True, max_chars=RETRIEVAL_HINT_CHARS)
    retrieval_query = last_user_msg if not hint else f"{last_user_msg}\n\n[History hint]\n{hint}"

    chat_row = await get_chat_row(session, chat_id)

    # 1) Recuperación texto+tabla
    hits_text_tbl = await vector_search_text_and_tables(session, chat_id, retrieval_query, top_k=top_k, min_score=0.0)

    # 2) Recuperación imágenes (candidatas) + filtrado por score relativo
    hits_img_all = await vector_search_img_desc(session, chat_id, retrieval_query, top_k=RAG_IMG_MIX_K)
    worst_text_score = min([float(h.get("score") or 0.0) for h in hits_text_tbl] + [1.0]) if hits_text_tbl else 1.0
    rel_threshold = max(IMG_MIN_SCORE, worst_text_score - IMG_SCORE_MARGIN)
    hits_img_qual = [h for h in hits_img_all if float(h.get("score") or 0.0) >= rel_threshold]
    hits_img_qual = hits_img_qual[:RAG_IMG_MIX_LIMIT]

    # 3) Merge + dedupe
    seen = set()
    merged: List[Dict[str, Any]] = []
    for h in hits_text_tbl + hits_img_qual:
        if h["id"] in seen:
            continue
        seen.add(h["id"])
        merged.append(h)

    # 4) doc_summary primero (si existe)
    try:
        row = (await session.execute(sqltext("""
            SELECT id, kind, content, page, meta
            FROM public.chunks
            WHERE chat_id = :chat
              AND kind = 'text'
              AND (meta->>'kind') = 'doc_summary'
            ORDER BY created_at DESC
            LIMIT 1
        """), {"chat": str(chat_id)})).mappings().first()
        if row:
            meta = row["meta"] if isinstance(row["meta"], dict) else json.loads(row["meta"] or "{}")
            merged = [{
                "id": str(row["id"]),
                "kind": "text",
                "content": row["content"] or "",
                "page": row["page"],
                "meta": meta,
                "score": 1.100,
            }] + merged
    except Exception:
        pass

    # 5) Orden final
    merged.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)

    # 6) Contexto RAG
    ctx_str = _make_context_string(merged, MAX_SOURCE_CHARS)

    # 7) Mensajes al LLM (core + persona + <document_context>)
    system_blocks = _compose_system_messages(
        chat_row,
        persona_prompt=system_prompt,
        prompt_mode=prompt_mode
    )
    oa_msgs: List[Dict[str, Any]] = [{"role": "system", "content": b} for b in system_blocks]

    if ctx_str:
        oa_msgs.append({"role": "system", "content": f"<document_context>\n{ctx_str}\n</document_context>"})

    # historial + último user
    hist_no_last = history_window[:-1] if (history_window and history_window[-1].get("role")=="user") else history_window
    oa_msgs.extend(hist_no_last)
    oa_msgs.append({"role": "user", "content": last_user_msg})

    # 8) Selección de modelo/temperatura/tokens (precedencia: param -> chat_row -> defaults)
    effective_model = (model or (chat_row.get("model") or ANSWER_TEXT_MODEL or OPENAI_MODEL_FALLBACK)).strip()
    effective_temperature = (
        DEFAULT_RAG_TEMPERATURE
        if temperature is None and chat_row.get("temperature") is None
        else float(temperature if temperature is not None else (chat_row.get("temperature") or DEFAULT_RAG_TEMPERATURE))
    )
    effective_max_tokens = int(max_tokens if max_tokens is not None else (chat_row.get("max_tokens") or 700))

    # 9) Llamada al LLM
    resp = _client.chat.completions.create(
        model=effective_model,
        messages=oa_msgs,
        temperature=effective_temperature,
        max_tokens=effective_max_tokens,
    )
    answer = (resp.choices[0].message.content or "").strip()

    # 10) Fuentes para UI
    sources = []
    for h in merged:
        meta = h.get("meta") or {}
        item = {
            "chunk_id": h["id"],
            "kind": h["kind"],   # 'text' | 'table' | 'img_text'
            "page": h.get("page"),
            "score": h["score"],
        }
        if h["kind"] == "table":
            item["html"] = h.get("content") or ""
            if meta.get("plain"):
                item["plain_preview"] = _trim(meta["plain"], 400)
            if meta.get("bbox"):       item["bbox"] = meta["bbox"]
            if meta.get("bbox_norm"):  item["bbox_norm"] = meta["bbox_norm"]
            if meta.get("order_index") is not None:
                item["order_index"] = meta["order_index"]
        elif h["kind"] == "img_text":
            item["preview"] = _trim(h.get("content") or "", 400)
            item["image_uid"] = meta.get("image_uid")
            if meta.get("bbox"):       item["bbox"] = meta["bbox"]
            if meta.get("bbox_norm"):  item["bbox_norm"] = meta["bbox_norm"]
            if meta.get("order_index") is not None:
                item["order_index"] = meta["order_index"]
            if meta.get("w"):          item["w"] = meta["w"]
            if meta.get("h"):          item["h"] = meta["h"]
            if meta.get("dpi_used"):   item["dpi_used"] = meta["dpi_used"]
            if meta.get("tags"):       item["tags"] = meta["tags"]
            if meta.get("objects"):    item["objects"] = meta["objects"]
            if meta.get("colors"):     item["colors"] = meta["colors"]
        else:
            item["preview"] = _trim(h.get("content") or "", 400)
        sources.append(item)

    return {
        "answer": answer,
        "sources": sources,
        "used_top_k": top_k,
        "img_mix_k": len([s for s in sources if s["kind"] == "img_text"]),
        "model_used": effective_model,
        "temperature_used": effective_temperature,
        "max_tokens_used": effective_max_tokens,
        "prompt_mode_used": (prompt_mode or "merge"),
        "system_messages_count": len(system_blocks) + (1 if ctx_str else 0),
    }