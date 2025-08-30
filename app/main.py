import sys, asyncio
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.base import Base
from app.db.session import engine

from app.routers import health
from app.routers import chats as chats_router
from app.routers import ingest as ingest_router
from app.routers import rag as rag_router   # <- NEW

load_dotenv()

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

app = FastAPI(title="Chatbase Clone API")

# CORS (ajustá orígenes en prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router)
app.include_router(chats_router.router)
app.include_router(ingest_router.router)
app.include_router(rag_router.router)        # <- NEW  (/rag/chats/{chat_id}/query, /rag/chats/{chat_id}/search)

# Opcional: ping rápido
@app.get("/")
async def root():
    return {"ok": True, "service": "chatbase-clone", "routers": ["health","chats","ingest","rag"]}

# Crear tablas si no existen (MVP). En prod, usar Alembic.
@app.on_event("startup")
async def on_startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)