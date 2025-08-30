import json, os, re, shutil, subprocess, tempfile, requests, copy
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID
from sqlalchemy import text as sqltext, bindparam
from sqlalchemy.types import String
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI

load_dotenv()

# ====== CONFIG ======
PDF_EXTRACTOR_PATH = os.getenv("PDF_EXTRACTOR_PATH", "app/scripts/pdf_extractor.py")
TABULA_JAR        = os.getenv("TABULA_JAR",        "app/scripts/tabula.jar")
DOCS_BUCKET       = os.getenv("DOCS_BUCKET",       "docs")

EMBED_MODEL       = os.getenv("EMBED_MODEL",       "text-embedding-3-small")
VISION_MODEL      = os.getenv("VISION_MODEL",      "gpt-4o-mini")   # modelo de visión barato por defecto
VISION_MAX_TOKENS = int(os.getenv("VISION_MAX_TOKENS", "400"))

SUPABASE_PROJECT_URL = os.environ.get("SUPABASE_PROJECT_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

if not SUPABASE_PROJECT_URL or not SUPABASE_SERVICE_KEY:
    print("[BOOT] WARNING: faltan SUPABASE_PROJECT_URL y/o SUPABASE_SERVICE_KEY en el entorno")

supabase: Client = create_client(SUPABASE_PROJECT_URL, SUPABASE_SERVICE_KEY)

# ====== OPENAI CLIENT ======
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ====== EMBEDDINGS ======
def vec_literal(vec: List[float]) -> str:
    """Formatea una lista de floats al literal pgvector: '[0.1,-0.2,...]'"""
    if not vec:
        return "[]"
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

async def embed_text(text: str) -> List[float]:
    t = (text or "").strip()
    if not t:
        # vector nulo del tamaño del modelo (1536 para text-embedding-3-small)
        return [0.0] * 1536
    resp = _client.embeddings.create(model=EMBED_MODEL, input=t)
    return resp.data[0].embedding

# ====== STORAGE: solo descarga del PDF ======
def _normalize_storage_key_from_public_url(url: str, expected_bucket: str) -> Optional[str]:
    try:
        parts = url.split("/")
        idx = parts.index("public")
        bucket = parts[idx + 1]
        key = "/".join(parts[idx + 2:])
        if bucket != expected_bucket:
            print(f"[STORAGE] WARN: bucket URL={bucket} != esperado={expected_bucket}")
            return None
        if key.startswith(f"{expected_bucket}/"):
            key = key[len(expected_bucket) + 1 :]
        if key.startswith("docs/docs/"):
            key = key[len("docs/") :]
        return key
    except Exception:
        return None

async def storage_file_exists(bucket: str, object_name: str) -> bool:
    """Verifica si un archivo existe en el storage"""
    try:
        if object_name.startswith("http://") or object_name.startswith("https://"):
            key = _normalize_storage_key_from_public_url(object_name, bucket)
            if key:
                object_name = key
        
        # Intentar obtener info del archivo
        file_list = supabase.storage.from_(bucket).list(path="/".join(object_name.split("/")[:-1]) or "")
        file_name = object_name.split("/")[-1]
        return any(f["name"] == file_name for f in file_list)
    except Exception as e:
        print(f"[STORAGE] Error verificando existencia del archivo {object_name}: {e}")
        return False

async def storage_download_to_tmp(bucket: str, object_name: str, dst_path: Path) -> None:
    print(f"[STORAGE] download: bucket={bucket}, object_name={object_name}, dst={dst_path}")
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if object_name.startswith("http://") or object_name.startswith("https://"):
        key = _normalize_storage_key_from_public_url(object_name, bucket)
        if key:
            print(f"[STORAGE] usando SDK con key normalizada: {key}")
            data: bytes = supabase.storage.from_(bucket).download(key)
            if not data:
                raise RuntimeError(f"Storage download vacío (SDK) para {object_name}")
            dst_path.write_bytes(data)
            print(f"[STORAGE] SDK OK ({len(data)} bytes)")
            return

        print(f"[STORAGE] HTTP GET directo: {object_name}")
        r = requests.get(object_name, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP GET failed {r.status_code} para {object_name}")
        dst_path.write_bytes(r.content)
        print(f"[STORAGE] HTTP OK ({len(r.content)} bytes)")
        return

    try:
        data: bytes = supabase.storage.from_(bucket).download(object_name)
        if not data:
            raise RuntimeError(f"Storage download vacío para {object_name}. Posibles causas: archivo no existe, permisos insuficientes, o bucket incorrecto.")
        dst_path.write_bytes(data)
        print(f"[STORAGE] SDK OK ({len(data)} bytes)")
    except Exception as e:
        raise RuntimeError(f"Error descargando desde storage - bucket: {bucket}, object: {object_name}, error: {str(e)}")

# ====== UTILS ======
def minify_html(html: str) -> str:
    return " ".join((html or "").split())

def html_to_plain(html: str, max_rows: int = 12) -> str:
    # headers + primeras filas a texto simple (para embedding)
    if not html:
        return ""
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.I | re.S)
    out = []
    for i, row in enumerate(rows):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.I | re.S)
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        if any(cells):
            out.append(" | ".join([c for c in cells if c]))
        if i >= max_rows:
            break
    return "\n".join(out)

def split_for_embedding(text: str, max_chars: int = 2400) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts, buf, count = [], [], 0
    for para in re.split(r"\n{2,}", text):
        p = para.strip()
        if not p:
            continue
        if count + len(p) + 2 > max_chars and buf:
            parts.append("\n\n".join(buf)); buf = [p]; count = len(p)
        else:
            buf.append(p); count += len(p) + 2
    if buf:
        parts.append("\n\n".join(buf))
    return parts

# Marcadores a limpiar del texto de página
IMG_TAG  = re.compile(r'\[IMAGE:[^\]]+\]')
IMG_LINE = re.compile(r'^<image:[^>]+>\s*$', re.MULTILINE)

def clean_text_for_embed(s: str) -> str:
    """Quita marcadores de imagen y normaliza espacios para mejorar el embedding."""
    s = IMG_TAG.sub('', s or '')
    s = IMG_LINE.sub('', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# ====== VISIÓN (anotar una imagen a JSON) ======
def vision_annotate_image(data_url: str) -> Dict[str, Any]:
    """
    Devuelve JSON:
    {
      caption: str,
      tags: string[],
      colors: {dominant:string[], palette:string[], dominant_hex:string[]},
      ocr_text: string,
      objects: string[],
      counts: object,
      approx_size: string|null,
      detected_languages: string[],
      safety: {nsfw:boolean, note:string|null}
    }
    """
    system_prompt = (
        "You are an assistant that produces STRICT JSON about an image. "
        "Return ONLY JSON, no prose. Keys: "
        "{caption:string, tags:string[], colors:{dominant:string[], palette:string[], dominant_hex:string[]}, "
        "ocr_text:string, objects:string[], counts:object, approx_size:string|null, "
        "detected_languages:string[], safety:{nsfw:boolean, note:string|null}}. "
        "caption: concise and information-rich; 5-12 tags; colors as common names and hex if clear; "
        "ocr_text: short readable text; objects: nouns/brands/models; "
        "counts: e.g. {'screws':3}; approx_size if inferable (e.g. 'hand-sized'), else null."
    )
    try:
        resp = _client.chat.completions.create(
            model=VISION_MODEL,
            temperature=0.2,
            max_tokens=VISION_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": "Analyze this image and produce ONLY the JSON."},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]}
            ]
        )
        text = (resp.choices[0].message.content or "").strip()
        text = text.strip("` \n")
        if text.startswith("json"):
            text = text[4:].strip(": \n`")
        data = json.loads(text)
    except Exception as e:
        print(f"[VISION] WARN parsing vision JSON: {repr(e)}")
        data = {
            "caption": "",
            "tags": [],
            "colors": {"dominant": [], "palette": [], "dominant_hex": []},
            "ocr_text": "",
            "objects": [],
            "counts": {},
            "approx_size": None,
            "detected_languages": [],
            "safety": {"nsfw": False, "note": None}
        }
    return data

# ====== MAIN SERVICE ======
async def ingest_source_service(session: AsyncSession, chat_id: UUID, source_id: UUID) -> Dict[str, Any]:
    print(f"[INGEST] Iniciando ingest para chat={chat_id}, source={source_id}")

    # 1) Validar source y traer file_url
    row = (await session.execute(sqltext("""
        select s.id as source_id, s.file_url, s.file_name, s.file_size, s.status,
               c.id as chat_id, c.user_id
          from public.knowledge_sources s
          join public.chats c on c.id = s.chat_id
         where s.id = :sid and c.id = :cid
    """), {"sid": str(source_id), "cid": str(chat_id)})).mappings().first()

    print(f"[INGEST] Validación DB: {row}")

    if not row:
        raise HTTPException(status_code=404, detail="source not found for this chat")
    if row["status"] != "processing":
        raise HTTPException(status_code=409, detail=f"invalid status {row['status']}")
    if not row["file_url"]:
        raise HTTPException(status_code=400, detail="source has no file_url")

    file_url  = row["file_url"]
    file_name = row["file_name"]
    file_size = row["file_size"]
    
    # Si file_name está vacío, extraerlo del file_url
    if not file_name and file_url:
        file_name = file_url.split('/')[-1] if '/' in file_url else file_url
        print(f"[INGEST] file_name extraído de URL: {file_name}")
    
    print(f"[INGEST] Archivo a procesar: {file_name} ({file_size} bytes) en {file_url}")

    # 2) Carpeta temporal
    tmpdir = Path(tempfile.mkdtemp(prefix="ing_"))
    pdf_local = tmpdir / "input.pdf"
    out_json = tmpdir / "out.json"
    print(f"[INGEST] Carpeta temporal creada: {tmpdir}")

    try:
        # 3) Verificar que el archivo existe y descargarlo
        print(f"[INGEST] Verificando existencia del archivo en bucket={DOCS_BUCKET}, key={file_url}")
        
        # Verificar si el archivo existe antes de intentar descargarlo
        if not await storage_file_exists(DOCS_BUCKET, file_url):
            error_msg = f"El archivo no existe en el storage: {file_url}"
            print(f"[INGEST] {error_msg}")
            await session.execute(sqltext("""
                update public.knowledge_sources
                   set status='failed', error_message=:msg, updated_at=now()
                 where id=:sid
            """), {"sid": str(source_id), "msg": error_msg})
            await session.commit()
            raise HTTPException(status_code=404, detail=error_msg)
        
        print(f"[INGEST] Descargando PDF desde bucket={DOCS_BUCKET}, key={file_url}")
        try:
            await storage_download_to_tmp(DOCS_BUCKET, file_url, pdf_local)
            actual_file_size = pdf_local.stat().st_size
            print(f"[INGEST] PDF descargado en {pdf_local} ({actual_file_size} bytes)")
        except Exception as storage_error:
            error_msg = f"Error descargando archivo: {str(storage_error)}"
            print(f"[INGEST] {error_msg}")
            await session.execute(sqltext("""
                update public.knowledge_sources
                   set status='failed', error_message=:msg, updated_at=now()
                 where id=:sid
            """), {"sid": str(source_id), "msg": error_msg})
            await session.commit()
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Si file_size es null, usar el tamaño real del archivo descargado
        if file_size is None:
            file_size = actual_file_size
            print(f"[INGEST] file_size actualizado desde archivo descargado: {file_size} bytes")

        # 4) Ejecutar extractor (texto + tablas HTML + imágenes como data_url)
        cmd = [
            "python", PDF_EXTRACTOR_PATH,
            "--pdf", str(pdf_local),
            "--out", str(out_json),
            "--dpi", "300",
            "--tabula-jar", TABULA_JAR,
            "--lattice"
        ]
        print(f"[INGEST] Ejecutando extractor: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        print(f"[INGEST] Extractor terminó code={proc.returncode}")
        if proc.stdout:
            print(f"[INGEST] stdout: {proc.stdout[:300]}...")
        if proc.stderr:
            print(f"[INGEST] stderr: {proc.stderr[:300]}...")

        if proc.returncode != 0 or not out_json.exists():
            await session.execute(sqltext("""
                update public.knowledge_sources
                   set status='failed', error_message=:msg, updated_at=now()
                 where id=:sid
            """), {"sid": str(source_id), "msg": f"extractor failed: {proc.stderr[:400]}"} )
            await session.commit()
            raise HTTPException(status_code=500, detail="extractor failed")

        data   = json.loads(out_json.read_text("utf-8"))
        pages  = data.get("pages", []) or []
        tables = data.get("tables", []) or []
        images = data.get("images", []) or []
        meta   = data.get("meta", {}) or {}

        print(f"[INGEST] Extraído: {len(pages)} páginas, {len(tables)} tablas, {len(images)} imágenes")

        # Copia del HTML original de tablas antes de minificar
        original_tables_by_id: Dict[str, str] = {}
        for _t in copy.deepcopy(tables):
            tid = _t.get("id")
            if tid:
                original_tables_by_id[tid] = _t.get("html") or ""

        # 5) CHUNKS + EMBEDDINGS
        total_chars = 0

        # 5.a TEXT (páginas) → limpiar marcadores e indexar
        print("[INGEST] Procesando TEXT de páginas...")
        for p in pages:
            page_n = p.get("n")
            txt = (p.get("text_with_marks") or "").strip()
            if not txt:
                continue
            for piece in split_for_embedding(txt):
                piece_clean = clean_text_for_embed(piece)
                if not piece_clean:
                    continue
                emb_list = await embed_text(piece_clean)
                emb_str  = vec_literal(emb_list)
                total_chars += len(piece_clean)

                stmt_text = sqltext("""
                    insert into public.chunks
                    (chat_id, source_id, kind, content, page, meta, embedding)
                    values
                    (:chat, :src, 'text', :content, :page, '{}'::jsonb,
                     CAST(:emb AS text)::vector)
                """).bindparams(bindparam("emb", type_=String))

                await session.execute(stmt_text, {
                    "chat": str(chat_id),
                    "src": str(source_id),
                    "content": piece_clean,
                    "page": page_n,
                    "emb": emb_str
                })

        # 5.b TABLES → guardar HTML en content + embedding de meta.plain
        print("[INGEST] Procesando TABLES...")
        for t in tables:
            tid = t.get("id")
            page_n = t.get("page")
            original_html = original_tables_by_id.get(tid, "")
            html_min = minify_html(original_html)
            plain    = html_to_plain(original_html)
            emb_str  = vec_literal(await embed_text(plain)) if plain else None
            total_chars += len(plain or "")

            meta_obj = {
                "table_id": tid,
                "bbox": t.get("bbox"),
                "bbox_norm": t.get("bbox_norm"),
                "order_index": t.get("order_index"),
                "plain": plain
            }

            stmt_tbl = sqltext("""
                insert into public.chunks
                (chat_id, source_id, kind, content, page, meta, embedding)
                values
                (:chat, :src, 'table', :content, :page, CAST(:meta AS jsonb),
                 CASE WHEN :emb IS NULL THEN NULL ELSE CAST(:emb AS text)::vector END)
            """).bindparams(bindparam("meta", type_=String),
                            bindparam("emb", type_=String))

            await session.execute(stmt_tbl, {
                "chat": str(chat_id),
                "src": str(source_id),
                "content": html_min,        # HTML completo de la tabla
                "page": page_n,
                "meta": json.dumps(meta_obj, ensure_ascii=False),
                "emb": emb_str
            })

        # 5.c IMAGES → visión barata → descriptor textual (chunk 'img_text' + meta rica)
        print("[INGEST] Procesando IMAGES (visión barata → texto)...")

        # orden estable por página (lectura natural: y0 asc, x0 asc)
        from collections import defaultdict
        imgs_by_page: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for im in images:
            page_n = int(im.get("page") or 0)
            bbox = im.get("bbox") or [0, 0, 0, 0]
            x0, y0 = float(bbox[0]), float(bbox[1])
            imgs_by_page[page_n].append((y0, x0, im))

        # ordenar y asignar order_index (fallback si el extractor no lo puso)
        for page_n, lst in imgs_by_page.items():
            lst.sort(key=lambda t: (t[0], t[1]))
            for idx, (_, __, im) in enumerate(lst, start=1):
                if im.get("order_index") is None:
                    im["order_index"] = idx
            imgs_by_page[page_n] = [t[2] for t in lst]  # reemplazar por lista de dicts

        total_images_inserted = 0
        per_page_counts: Dict[int, int] = {}

        for page_n in sorted(imgs_by_page.keys()):
            page_imgs = imgs_by_page[page_n]
            per_page_counts[page_n] = len(page_imgs)
            for im in page_imgs:
                data_url = im.get("data_url") or ""
                ann = vision_annotate_image(data_url)

                # construir texto embebible (ES+EN para mejor recall)
                parts = []
                header = f"[Imagen | Image] pág. {page_n} #{im.get('order_index')}"
                parts.append(header)

                if ann.get("tags"):
                    parts.append("Etiquetas/Tags: " + ", ".join((ann.get("tags") or [])[:12]))
                if ann.get("colors"):
                    dom = ann["colors"].get("dominant") or []
                    if dom:
                        parts.append("Colores/Colors: " + ", ".join(dom[:6]))
                if ann.get("objects"):
                    parts.append("Objetos/Objects: " + ", ".join((ann.get("objects") or [])[:12]))
                if ann.get("ocr_text"):
                    parts.append("Texto visible / Visible text: " + (ann.get("ocr_text") or "")[:600])
                if ann.get("caption"):
                    parts.append("Descripción / Caption: " + ann["caption"])
                if ann.get("approx_size"):
                    parts.append(f"Tamaño aprox.: {ann['approx_size']}")

                vision_text = "\n".join(parts).strip()
                emb_str = vec_literal(await embed_text(vision_text)) if vision_text else None
                total_chars += len(vision_text or "")

                vision_meta = {
                    "kind": "img_desc",
                    "image_uid": im.get("id"),
                    "page": page_n,
                    "bbox": im.get("bbox"),
                    "bbox_norm": im.get("bbox_norm"),
                    "order_index": im.get("order_index"),
                    "w": im.get("w"),
                    "h": im.get("h"),
                    "dpi_used": im.get("dpi_used"),
                    "image_md5": im.get("image_md5"),
                    "xref": im.get("xref"),
                    "tags": ann.get("tags", []),
                    "objects": ann.get("objects", []),
                    "colors": ann.get("colors", {}),
                    "ocr_text": ann.get("ocr_text", ""),
                    "approx_size": ann.get("approx_size"),
                    "detected_languages": ann.get("detected_languages", []),
                    "safety": ann.get("safety", {})
                }

                stmt_img = sqltext("""
                    insert into public.chunks
                    (chat_id, source_id, kind, content, page, meta, embedding)
                    values
                    (:chat, :src, 'img_text', :content, :page, CAST(:meta AS jsonb),
                     CASE WHEN :emb IS NULL THEN NULL ELSE CAST(:emb AS text)::vector END)
                """).bindparams(bindparam("meta", type_=String),
                                bindparam("emb", type_=String))

                await session.execute(stmt_img, {
                    "chat": str(chat_id),
                    "src": str(source_id),
                    "content": vision_text or "",
                    "page": page_n,
                    "meta": json.dumps(vision_meta, ensure_ascii=False),
                    "emb": emb_str
                })

                total_images_inserted += 1

        # 5.d IMG MANIFEST → un chunk 'text' con meta.kind='img_manifest'
        if total_images_inserted > 0:
            # resumen por página ordenado
            per_page_summary = ", ".join(
                f"{p}:{per_page_counts.get(p,0)}" for p in sorted(per_page_counts.keys())
            )
            manifest_text = (
                "[Resumen de imágenes del documento]\n"
                f"Total de imágenes: {total_images_inserted}\n"
                f"Por página: {per_page_summary}"
            )
            emb_str = vec_literal(await embed_text(manifest_text))
            total_chars += len(manifest_text)

            await session.execute(sqltext("""
                insert into public.chunks
                  (chat_id, source_id, kind, content, page, meta, embedding)
                values
                  (:chat, :src, 'text', :content, 1,
                   '{"kind":"img_manifest"}'::jsonb,
                   CAST(:emb AS text)::vector)
            """), {
                "chat": str(chat_id),
                "src": str(source_id),
                "content": manifest_text,
                "emb": emb_str
            })

        # 6) Guardar manifest "liviano" en knowledge_sources.content (opcional pero útil)
        light_manifest = {
            "original": {
                "bucket": DOCS_BUCKET,
                "key": file_url,
                "filename": file_name,
                "size": file_size
            },
            "meta": {
                "doc_type": meta.get("doc_type", "pdf"),
                "total_pages": meta.get("total_pages"),
                "ocr_used": meta.get("ocr_used", False),
                "page_sizes": meta.get("page_sizes"),
                "version": 1
            },
            "stats": {
                "pages": len(pages),
                "tables": len(tables),
                "images": len(images),
                "images_indexed": total_images_inserted
            },
            # NO guardamos data_url ni html crudo; solo IDs y bbox por si hace falta depurar
            "tables_brief": [
                {
                    "id": t.get("id"),
                    "page": t.get("page"),
                    "bbox": t.get("bbox"),
                    "bbox_norm": t.get("bbox_norm"),
                    "order_index": t.get("order_index"),
                }
                for t in tables
            ],
            "images_brief": [
                {
                    "id": im.get("id"),
                    "page": im.get("page"),
                    "bbox": im.get("bbox"),
                    "bbox_norm": im.get("bbox_norm"),
                    "order_index": im.get("order_index"),
                }
                for im in images
            ]
        }

        await session.execute(sqltext("""
            update public.knowledge_sources
               set status='active',
                   error_message=null,
                   character_count=:cc,
                   content=:content,
                   file_name=:file_name,
                   file_size=:file_size,
                   updated_at=now()
             where id=:sid
        """), {
            "sid": str(source_id),
            "cc": int(total_chars),
            "content": json.dumps(light_manifest, ensure_ascii=False),
            "file_name": file_name,
            "file_size": file_size
        })
        await session.commit()

        print(f"[INGEST] Ingest finalizado OK. chars={total_chars}")

        return {
            "pages": len(pages),
            "tables": len(tables),
            "images": len(images),
            "images_indexed": total_images_inserted,
            "characters": total_chars
        }

    except Exception as e:
        print(f"[INGEST] ERROR: {e}")
        await session.execute(sqltext("""
            update public.knowledge_sources
               set status='failed', error_message=:msg, updated_at=now()
             where id=:sid
        """), {"sid": str(source_id), "msg": str(e)[:500]})
        await session.commit()
        raise
    finally:
        print(f"[INGEST] Limpiando tmpdir {tmpdir}")
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass

# ====== TEXT SOURCE FROM DB ======
async def ingest_text_source_from_db(session: AsyncSession, *, chat_id: UUID, source_id: UUID):
    row = (await session.execute(sqltext("""
        select name, content, status
          from public.knowledge_sources
         where id = :sid and chat_id = :cid and type = 'text'
         limit 1
    """), {"sid": str(source_id), "cid": str(chat_id)})).mappings().first()

    if not row:
        raise HTTPException(status_code=404, detail="text source not found")
    content = (row["content"] or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="text source has empty content")

    # Limpio chunks previos para reindex (útil si se edita)
    await session.execute(sqltext("delete from public.chunks where source_id = :sid"), {"sid": str(source_id)})
    await session.execute(sqltext("""
        update public.knowledge_sources
           set status='processing', character_count = length(:c), updated_at=now()
         where id = :sid
    """), {"sid": str(source_id), "c": content})
    await session.commit()

    return await ingest_text_source(
        session,
        chat_id=str(chat_id),
        source_id=str(source_id),
        name=row["name"] or "Text snippet",
        content=content,
    )

# ====== TEXT-ONLY SOURCE ======
async def ingest_text_source(session: AsyncSession, *, chat_id: str, source_id: str, name: str, content: str):
    try:
        total_chars = len(content or "")
        print(f"[INGEST/TEXT] chars={total_chars}")

        for piece in split_for_embedding(content or ""):
            piece_clean = clean_text_for_embed(piece)
            if not piece_clean:
                continue
            emb_list = await embed_text(piece_clean)
            emb_str  = vec_literal(emb_list)
            await session.execute(sqltext("""
                insert into public.chunks
                  (chat_id, source_id, kind, content, page, meta, embedding)
                values
                  (:chat, :src, 'text', :content, 1, '{}'::jsonb,
                   CAST(:emb AS text)::vector)
            """), {
                "chat": str(chat_id),
                "src": str(source_id),
                "content": piece_clean,
                "emb": emb_str
            })

        await session.execute(sqltext("""
            update public.knowledge_sources
               set status='active',
                   error_message=null,
                   character_count=:cc,
                   updated_at=now()
             where id=:sid
        """), {"sid": str(source_id), "cc": total_chars})

        await session.commit()
        print("[INGEST/TEXT] OK")
        return {"characters": total_chars}

    except Exception as e:
        print(f"[INGEST/TEXT] ERROR: {e}")
        await session.execute(sqltext("""
            update public.knowledge_sources
               set status='failed', error_message=:msg, updated_at=now()
             where id=:sid
        """), {"sid": str(source_id), "msg": str(e)[:500]})
        await session.commit()
        raise

# ====== ENTRYPOINT PARA FILE ======
async def ingest_file_source(session: AsyncSession, *, chat_id: str, source_id: str):
    return await ingest_source_service(session=session, chat_id=UUID(chat_id), source_id=UUID(source_id))