from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_session
from app.middlewares.auth import current_user
from app.schemas.bot import BotCreate, BotOut
from app.services.tenants import get_or_create_default_tenant
from app.services.bots import create_bot

router = APIRouter(prefix="/bots", tags=["bots"])

@router.post("", response_model=BotOut)
async def create_bot_route(
    payload: BotCreate,
    user = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    tenant = await get_or_create_default_tenant(session, owner_user_id=user["sub"])
    bot = await create_bot(session, tenant_id=tenant.id, name=payload.name, system_prompt=payload.system_prompt)
    return {"bot_id": bot.id}