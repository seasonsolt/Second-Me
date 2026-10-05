from typing import List, Optional, Dict
from sqlalchemy import select, func, text
from lpm_kernel.common.repository.base_repository import BaseRepository
from lpm_kernel.file_data.document import Document
from lpm_kernel.file_data.process_status import ProcessStatus
from lpm_kernel.file_data.document_dto import DocumentDTO
from lpm_kernel.file_data.models import ChunkModel
from .dto.chunk_dto import ChunkDTO
import logging

logger = logging.getLogger(__name__)


class DocumentRepository(BaseRepository[Document]):
    def __init__(self):
        super().__init__(Document)

    def update_document_analysis(
        self, doc_id: int, insight: Dict, summary: Dict
    ) -> Optional[DocumentDTO]:
        """update doc's insight and summary"""
        with self._db.session() as session:
            document = session.get(self.model, doc_id)
            if document:
                document.insight = insight
                document.summary = summary
                document.analyze_status = ProcessStatus.SUCCESS
                session.commit()
                return Document.to_dto(document)
            return None

    def find_unanalyzed(self) -> List[DocumentDTO]:
        """search unanalyzed doc according to analyze_status"""
        with self._db.session() as session:
            query = select(self.model).where(
                self.model.analyze_status.in_([ProcessStatus.INITIALIZED, ProcessStatus.FAILED])
            )
            result = session.execute(query)
            return [Document.to_dto(doc) for doc in result.scalars().all()]

    def find_chunks(self, document_id: int) -> List[ChunkDTO]:
        """search all chunks of the specified document"""
        with self._db.session() as session:
            chunks = (
                session.query(ChunkModel)
                .filter(ChunkModel.document_id == document_id)
                .all()
            )
            return [
                ChunkDTO(
                    id=chunk.id,
                    document_id=chunk.document_id,
                    has_embedding=chunk.has_embedding,
                    # embedding=chunk.embedding,
                    length=len(chunk.content) if chunk.content else 0,
                    content=chunk.content,
                    tags=chunk.tags,
                    topic=chunk.topic,
                )
                for chunk in chunks
            ]

    def save_chunk(self, chunk: ChunkModel) -> ChunkModel:
        """save chunk"""
        with self._db.session() as session:
            if session.bind.dialect.name == "sqlite" and chunk.id is None:
                # Support existing BIGINT tables without a destructive migration.
                session.execute(text("BEGIN IMMEDIATE"))
                chunk.id = (session.query(func.max(ChunkModel.id)).scalar() or 0) + 1
            session.add(chunk)
            session.flush()  # get auto-gen ID
            session.refresh(chunk)
            return chunk

    def find_one(self, document_id: int) -> Optional[DocumentDTO]:
        """search doc by id"""
        with self._db.session() as session:
            document = session.get(self.model, document_id)
            return Document.to_dto(document) if document else None

    def update_chunk_embedding_status(self, chunk_id: int, has_embedding: bool) -> None:
        """update chunk embedding"""
        try:
            with self._db.session() as session:
                chunk = (
                    session.query(ChunkModel).filter(ChunkModel.id == chunk_id).first()
                )
                if chunk:
                    chunk.has_embedding = has_embedding
                    session.commit()
                    logger.debug(f"Updated embedding status for chunk {chunk_id}")
                else:
                    logger.warning(f"Chunk not found with id: {chunk_id}")
        except Exception as e:
            logger.error(f"Error updating chunk embedding status: {str(e)}")
            raise

    def find_unembedding(self) -> List[DocumentDTO]:
        """search unembedding documents according to embedding_status"""
        with self._db.session() as session:
            query = select(self.model).where(
                self.model.embedding_status.in_([ProcessStatus.INITIALIZED, ProcessStatus.FAILED])
            )
            result = session.execute(query)
            return [Document.to_dto(doc) for doc in result.scalars().all()]

    def update_embedding_status(self, document_id: int, status: ProcessStatus) -> None:
        """update doc embedding"""
        try:
            with self._db.session() as session:
                document = (
                    session.query(self.model)
                    .filter(self.model.id == document_id)
                    .first()
                )
                if document:
                    document.embedding_status = status.value
                    session.commit()
                    logger.debug(
                        f"Updated embedding status for document {document_id} to {status.value}"
                    )
                else:
                    logger.warning(f"Document not found with id: {document_id}")
        except Exception as e:
            logger.error(f"Error updating document embedding status: {str(e)}")
            raise
