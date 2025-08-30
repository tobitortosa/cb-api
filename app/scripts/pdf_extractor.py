#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF → JSON con:
- pages[].text_with_marks   (incluye [IMAGE:id] y [TABLE:id|html])
- tables[] {id, page, html, bbox, bbox_norm?, crop_data_url?, order_index?}
- images[] {id, page, bbox, bbox_norm?, data_url, w?, h?, dpi_used?, image_md5?, xref?, order_index?}
- meta: {doc_type, ocr_used, total_pages, used_pages_count, page_sizes:[{page,width,height,rotation}]}

Uso:
  python doc_layout_extractor.py --pdf input.pdf --out out.json \
      --tabula-jar path/a/tabula.jar --lattice --dpi 300 --table-crops \
      --pages "1-3,5"
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ==== Logging ====
LOG_FORMAT = "[%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
log = logging.getLogger("doc_layout_extractor")

# ==== Dependencias duras ====
try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF no instalado. Ejecutá: pip install pymupdf", file=sys.stderr)
    sys.exit(1)

# ==== Utilidades ====

def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
    )

def _data_url_from_pixmap(pix: "fitz.Pixmap", mime: str = "image/png") -> Tuple[str, bytes]:
    """
    Devuelve (data_url, raw_bytes_codificados) para poder calcular MD5 si hace falta.
    """
    fmt = "png" if mime.endswith("png") else "jpeg"
    raw = pix.tobytes(fmt)  # bytes ya comprimidos en PNG/JPEG
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}", raw

@dataclass
class Block:
    y: float
    x: float
    kind: str  # "text" | "image" | "table"
    payload: Dict[str, Any]

# ==== Extracciones ====

def extract_text_blocks(doc: "fitz.Document", page_indices: List[int]) -> Dict[int, List[Block]]:
    per_page: Dict[int, List[Block]] = {}
    for p in page_indices:
        page = doc[p]
        blocks = page.get_text("blocks")  # [(x0,y0,x1,y1,text, ...)]
        items: List[Block] = []
        for b in blocks:
            x0, y0, x1, y1, text, *_ = b
            text = (text or "").strip()
            if text:
                items.append(Block(y=float(y0), x=float(x0), kind="text", payload={"text": text}))
        per_page[p] = items
    return per_page

def extract_images(doc: "fitz.Document", page_indices: List[int], dpi: int = 300, max_per_page: Optional[int] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in page_indices:
        page = doc[p]
        pw, ph = float(page.rect.width), float(page.rect.height)
        found = 0
        for info in page.get_images(full=True):
            # info: (xref, smask, width, height, bpc, colorspace, alt, name, filter, ... )
            xref = info[0] if len(info) > 0 else 0
            rects = []
            try:
                if xref:
                    rects = page.get_image_rects(xref)
            except Exception:
                rects = []
            if not rects:
                continue
            for rect in rects:
                if max_per_page and found >= max_per_page:
                    break
                iid = f"img-{uuid.uuid4().hex}"
                mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
                pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
                data_url, raw_bytes = _data_url_from_pixmap(pix, mime="image/png")
                md5 = hashlib.md5(raw_bytes).hexdigest()

                bbox = [rect.x0, rect.y0, rect.x1, rect.y1]
                bbox_norm = [
                    (rect.x0 / pw) if pw else 0.0,
                    (rect.y0 / ph) if ph else 0.0,
                    (rect.x1 / pw) if pw else 0.0,
                    (rect.y1 / ph) if ph else 0.0,
                ]
                out.append({
                    "id": iid,
                    "page": p + 1,
                    "bbox": bbox,
                    "bbox_norm": bbox_norm,
                    "data_url": data_url,
                    "w": int(pix.width),
                    "h": int(pix.height),
                    "dpi_used": int(dpi),
                    "image_md5": md5,
                    "xref": int(xref)
                })
                found += 1
    return out

def _run_tabula(pdf_path: str, pages_spec: Optional[str], lattice: bool, tabula_jar: str) -> List[Dict[str, Any]]:
    args = [
        "java", "-Dfile.encoding=UTF-8", "-jar", tabula_jar,
        "-f", "JSON",
        "-p", pages_spec or "all",
    ]
    if lattice:
        args.append("-l")
    args.append(pdf_path)

    try:
        proc = subprocess.run(args, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        log.warning("Java no encontrado en PATH. Si no necesitás tablas, omití --tabula-jar.")
        return []

    if proc.returncode != 0:
        log.warning("Tabula error (%s): %.300s", proc.returncode, proc.stderr)
        return []

    try:
        data = json.loads(proc.stdout)
        if not isinstance(data, list):
            log.warning("Tabula devolvió una salida no lista")
            return []
        return data
    except json.JSONDecodeError:
        log.warning("No se pudo parsear la salida de Tabula")
        return []

def _build_html_from_tabula_json(t: Dict[str, Any], merge_heuristics: bool = False) -> Tuple[str, Optional[List[float]]]:
    rows = t.get("data", [])
    left = t.get("left"); top = t.get("top"); width = t.get("width"); height = t.get("height")
    bbox = [float(left), float(top), float(left) + float(width), float(top) + float(height)] \
        if None not in (left, top, width, height) else None

    # Construcción básica de tabla (sin heurísticas complejas)
    html_parts: List[str] = ["<table>"]
    for row in rows:
        if not row or all(not c.get("text", "").strip() for c in row):
            continue
        html_parts.append("<tr>")
        for cell in row:
            text = (cell.get("text") or "").strip()
            text = _escape_html(text) if text else "&nbsp;"
            html_parts.append(f"<td>{text}</td>")
        html_parts.append("</tr>")
    html_parts.append("</table>")

    # Reconstrucción mínima si vino casi vacía
    if len(rows) <= 1:
        html_parts = ["<table>"]
        seen_texts = set()
        for row in rows or []:
            y_groups: Dict[float, List[Dict[str, Any]]] = {}
            for cell in row or []:
                if (cell.get("text") or "").strip():
                    y = float(cell.get("y", 0))
                    placed = False
                    for y_key in list(y_groups.keys()):
                        if abs(y - y_key) < 5:
                            y_groups[y_key].append(cell); placed = True; break
                    if not placed:
                        y_groups[y] = [cell]
            for y in sorted(y_groups.keys()):
                cells = sorted(y_groups[y], key=lambda c: float(c.get("x", 0)))
                html_parts.append("<tr>")
                for cell in cells:
                    text = (cell.get("text") or "").strip()
                    if text and text not in seen_texts:
                        seen_texts.add(text)
                        html_parts.append(f"<td>{_escape_html(text)}</td>")
                html_parts.append("</tr>")
        html_parts.append("</table>")

    return "".join(html_parts), bbox

def extract_tables_with_tabula(pdf_path: str,
                               pages_spec: Optional[str],
                               lattice: bool,
                               tabula_jar: Optional[str],
                               doc: Optional["fitz.Document"],
                               dpi: int,
                               include_crops: bool,
                               merge_heuristics: bool = False) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not tabula_jar or not Path(tabula_jar).exists():
        return out

    raw_tables = _run_tabula(pdf_path, pages_spec, lattice, tabula_jar)
    if not raw_tables:
        return out

    for t in raw_tables:
        html, bbox = _build_html_from_tabula_json(t, merge_heuristics=merge_heuristics)
        tid = f"tbl-{uuid.uuid4().hex}"
        page_num = int(t.get("page", 0))
        item: Dict[str, Any] = {"id": tid, "page": page_num, "html": html, "bbox": bbox}

        # Normalizar bbox si es posible
        if doc is not None and bbox is not None and 1 <= page_num <= len(doc):
            page = doc[page_num - 1]
            pw, ph = float(page.rect.width), float(page.rect.height)
            item["bbox_norm"] = [
                (bbox[0] / pw) if pw else 0.0,
                (bbox[1] / ph) if ph else 0.0,
                (bbox[2] / pw) if pw else 0.0,
                (bbox[3] / ph) if ph else 0.0,
            ]
            if include_crops:
                rect = fitz.Rect(*bbox)
                mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
                pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
                data_url, _raw = _data_url_from_pixmap(pix, mime="image/png")
                item["crop_data_url"] = data_url

        out.append(item)

    return out

# ==== Armado del texto con marcadores ====

def build_pages_with_marks(doc: "fitz.Document",
                           text_blocks: Dict[int, List[Block]],
                           images: List[Dict[str, Any]],
                           tables: List[Dict[str, Any]],
                           page_indices: List[int]) -> List[Dict[str, Any]]:
    pages_out: List[Dict[str, Any]] = []

    # Mapear imágenes/tablas por página como Blocks (y, x) para ordenar determinísticamente
    imgs_by_page: Dict[int, List[Block]] = {}
    for im in images:
        p = (im["page"] - 1)
        y = float(im["bbox"][1]) if im.get("bbox") else 0.0
        x = float(im["bbox"][0]) if im.get("bbox") else 0.0
        imgs_by_page.setdefault(p, []).append(Block(y=y, x=x, kind="image", payload=im))

    tbls_by_page: Dict[int, List[Block]] = {}
    for tb in tables:
        p = (tb["page"] - 1)
        y = float(tb["bbox"][1]) if tb.get("bbox") else 0.0
        x = float(tb["bbox"][0]) if tb.get("bbox") else 0.0
        tbls_by_page.setdefault(p, []).append(Block(y=y, x=x, kind="table", payload=tb))

    for p in page_indices:
        blocks: List[Block] = []
        blocks.extend(text_blocks.get(p, []))
        blocks.extend(imgs_by_page.get(p, []))
        blocks.extend(tbls_by_page.get(p, []))

        # Orden natural: primero por y, luego por x
        blocks.sort(key=lambda b: (b.y, b.x))

        # Asignar order_index por tipo dentro de la página (determinista)
        # Imágenes
        if p in imgs_by_page:
            imgs_by_page[p].sort(key=lambda b: (b.y, b.x))
            for idx, b in enumerate(imgs_by_page[p], start=1):
                b.payload["order_index"] = idx
        # Tablas
        if p in tbls_by_page:
            tbls_by_page[p].sort(key=lambda b: (b.y, b.x))
            for idx, b in enumerate(tbls_by_page[p], start=1):
                b.payload["order_index"] = idx

        # Construcción del texto con marcas
        lines: List[str] = []
        for b in blocks:
            if b.kind == "text":
                txt = b.payload.get("text", "").strip()
                if txt:
                    lines.append(txt)
            elif b.kind == "image":
                iid = b.payload.get("id")
                lines.append(f"[IMAGE:{iid}]")
            elif b.kind == "table":
                tid = b.payload.get("id")
                lines.append(f"[TABLE:{tid}|html]")

        pages_out.append({"n": p + 1, "text_with_marks": "\n".join(lines).strip()})

    return pages_out

# ==== CLI / Main ====

def parse_pages_spec(pages: Optional[str], max_pages: int) -> Tuple[List[int], Optional[str]]:
    if not pages or pages.lower() == "all":
        return list(range(max_pages)), None
    parts = [s.strip() for s in pages.split(',') if s.strip()]
    idxs: List[int] = []
    for part in parts:
        if '-' in part:
            a, b = part.split('-', 1)
            a = int(a); b = int(b)
            for i in range(a, b + 1):
                if 1 <= i <= max_pages:
                    idxs.append(i - 1)
        else:
            i = int(part)
            if 1 <= i <= max_pages:
                idxs.append(i - 1)
    return sorted(set(idxs)), pages

def main():
    ap = argparse.ArgumentParser(description="PDF → JSON (texto+marcas, tablas HTML, imágenes)")
    ap.add_argument('--pdf', required=True, help='Ruta al PDF de entrada')
    ap.add_argument('--out', default='out.json', help='Ruta del JSON de salida')
    ap.add_argument('--pages', default=None, help="Rango de páginas, ej: '1-3,5' (por defecto: todas)")
    ap.add_argument('--dpi', type=int, default=300, help='DPI para renders (imágenes y crops)')
    ap.add_argument('--max-images-per-page', type=int, default=None, help='Límite de imágenes por página')
    ap.add_argument('--tabula-jar', default=None, help='Ruta a tabula.jar (si no se indica, no hay tablas)')
    ap.add_argument('--lattice', action='store_true', help='Usar modo lattice en Tabula (bordes visibles)')
    ap.add_argument('--table-crops', action='store_true', help='Adjuntar crop de la tabla como data_url (png)')
    ap.add_argument('--merge-heuristics', action='store_true', help='Reconstrucción básica de filas/columnas si Tabula viene pobre')

    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        log.error("No existe el archivo: %s", pdf_path)
        sys.exit(2)

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    page_indices, pages_spec = parse_pages_spec(args.pages, total_pages)
    log.info("Páginas del documento: %s (usando %s páginas)", total_pages, len(page_indices))

    # 1) Texto
    text_blocks = extract_text_blocks(doc, page_indices)

    # 2) Imágenes
    images = extract_images(doc, page_indices, dpi=args.dpi, max_per_page=args.max_images_per_page)

    # 3) Tablas (si hay tabula.jar)
    tables = extract_tables_with_tabula(
        pdf_path=str(pdf_path),
        pages_spec=pages_spec,
        lattice=args.lattice,
        tabula_jar=args.tabula_jar,
        doc=doc,
        dpi=args.dpi,
        include_crops=args.table_crops,
        merge_heuristics=args.merge_heuristics,
    )

    # 4) Texto con marcas (asigna order_index para imágenes y tablas)
    pages_out = build_pages_with_marks(doc, text_blocks, images, tables, page_indices)

    # 5) Metadatos de páginas (útiles para overlays/normalización)
    page_sizes = []
    for p in page_indices:
        page = doc[p]
        page_sizes.append({
            "page": p + 1,
            "width": float(page.rect.width),
            "height": float(page.rect.height),
            "rotation": int(page.rotation)
        })

    # 6) Salida JSON
    out: Dict[str, Any] = {
        "pages": pages_out,
        "tables": tables,
        "images": images,
        "meta": {
            "doc_type": "pdf",
            "ocr_used": False,  # si luego agregás OCR, podés setearlo
            "total_pages": total_pages,
            "used_pages_count": len(page_indices),
            "page_sizes": page_sizes
        }
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    log.info("OK → %s", out_path)

if __name__ == '__main__':
    main()