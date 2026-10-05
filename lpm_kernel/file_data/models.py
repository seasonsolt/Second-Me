from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    ForeignKey,
    JSON,
    String,
    Text,
    Integer,
    Boolean,
    Enum as SQLAlchemyEnum,
)
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from .process_status import ProcessStatus
from .dto.chunk_dto import ChunkDTO


Base = declarative_base()


class ChunkModel(Base):
    __tablename__ = "chunk"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True)
    document_id = Column(BigInteger, ForeignKey("document.id"), nullable=False)
    content = Column(Text, nullable=False)
    has_embedding = Column(Boolean, default=False)
    tags = Column(JSON)
    topic = Column(String(255))
    create_time = Column(DateTime, default=datetime.utcnow)

    document = relationship("DocumentModel", back_populates="chunks")

    def to_dto(self) -> ChunkDTO:
        return ChunkDTO(
            id=self.id,
            document_id=self.document_id,
            content=self.content,
            has_embedding=self.has_embedding,  # ensure this field is correctly converted
            tags=self.tags,
            topic=self.topic,
            length=len(self.content) if self.content else 0,
        )


class DocumentModel(Base):
    __tablename__ = "document"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    title = Column(String(255))
    mime_type = Column(String(100))
    user_description = Column(Text)
    url = Column(String(1024))
    document_size = Column(Integer, default=0)
    raw_content = Column(Text)
    insight = Column(JSON)
    summary = Column(JSON)
    keywords = Column(JSON)
    extract_status = Column(
        SQLAlchemyEnum(ProcessStatus), default=ProcessStatus.INITIALIZED
    )
    embedding_status = Column(
        SQLAlchemyEnum(ProcessStatus), default=ProcessStatus.INITIALIZED
    )
    create_time = Column(DateTime, default=datetime.now)
    update_time = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    chunks = relationship(
        "ChunkModel", back_populates="document", cascade="all, delete-orphan"
    )
