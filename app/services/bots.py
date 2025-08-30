import re
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.models.bot import Bot
from app.utils.ids import new_id

def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "bot"

async def create_bot(session: AsyncSession, tenant_id: str, name: str, system_prompt: str | None) -> Bot:
    slug = slugify(name)
    # Si existe, agrega sufijo
    i = 1
    while (await session.execute(select(Bot).where(Bot.slug == slug))).scalar_one_or_none():
        i += 1
        slug = f"{slug}-{i}"

    bot = Bot(
        id=new_id("bot"),
        tenant_id=tenant_id,
        name=name,
        slug=slug,
        system_prompt=system_prompt or "Eres un agente útil, claro y conciso.",
        origin_whitelist="",
        is_published=False,
    )
    session.add(bot)
    await session.commit()
    await session.refresh(bot)
    return bot