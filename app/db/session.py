from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.core.config import settings

# Engine asincrónico
engine = create_async_engine(settings.database_url, echo=False, future=True)

# Session factory
AsyncSessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)

# Dependency para inyectar sesión en endpoints
async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
