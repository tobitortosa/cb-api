from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models.tenant import Tenant
from app.utils.ids import new_id

async def get_or_create_default_tenant(session: AsyncSession, owner_user_id: str) -> Tenant:
    q = await session.execute(select(Tenant).where(Tenant.owner_user_id == owner_user_id))
    t = q.scalar_one_or_none()
    if t: 
        return t
    t = Tenant(id=new_id("ten"), owner_user_id=owner_user_id, name="My workspace")
    session.add(t)
    await session.commit()
    await session.refresh(t)
    return t