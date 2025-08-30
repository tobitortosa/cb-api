from sqlalchemy import Column, String, DateTime, func
from app.db.base import Base

class Tenant(Base):
    __tablename__ = "tenants"
    id = Column(String, primary_key=True)          # ulid/uuid
    owner_user_id = Column(String, nullable=False) # sub de Supabase
    name = Column(String, default="Workspace")
    created_at = Column(DateTime(timezone=True), server_default=func.now())