from sqlalchemy import Column, String, Integer, DateTime, func, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from app.db.base import Base

class KnowledgeSource(Base):
    __tablename__ = "knowledge_sources"
    id = Column(UUID(as_uuid=True), primary_key=True)
    chat_id = Column(UUID(as_uuid=True), ForeignKey("chats.id", ondelete="CASCADE"), nullable=False)

    type = Column(String, nullable=False)        # 'file' | 'text' | 'website' | 'qa'
    name = Column(String, nullable=False)

    content = Column(String)                     # solo cuando type='text'
    file_url = Column(String)
    file_name = Column(String)
    file_size = Column(Integer)

    character_count = Column(Integer, default=0)
    status = Column(String, default="active")    # 'active' | 'processing' | 'failed' | 'disabled'
    error_message = Column(String)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now())