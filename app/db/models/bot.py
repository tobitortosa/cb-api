from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship
from app.db.base import Base

class Bot(Base):
    __tablename__ = "bots"
    id = Column(String, primary_key=True)          # ulid/uuid
    tenant_id = Column(String, ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    name = Column(String, nullable=False)
    slug = Column(String, unique=True, nullable=False)
    system_prompt = Column(String, nullable=False)
    origin_whitelist = Column(String, default="")  # csv simple en MVP
    is_published = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    tenant = relationship("Tenant")