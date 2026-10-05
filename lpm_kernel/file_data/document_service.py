# file_data/service.py
from pathlib import Path
from typing import List, Dict, Optional
import os
from sqlalchemy import select

from lpm_kernel.common.repository.database_session import DatabaseSession
from lpm_kernel.common.repository.vector_store_factory import VectorStoreFactory
from lpm_kernel.file_data.document_dto import DocumentDTO, CreateDocumentRequest
from lpm_kernel.file_data.exceptions import FileProcessingError
from lpm_kernel.kernel.l0_base import InsightKernel, SummaryKernel
from lpm_kernel.models.memory import Memory
from .document import Document
from .document_repository import DocumentRepository
from .dto.chunk_dto import ChunkDTO
from .embedding_service import EmbeddingService
from .process_factory import ProcessorFactory
from .process_status import ProcessStatus

from lpm_kernel.configs.logging import get_train_process_logger
logger = get_train_process_logger()


class DocumentService:
    def __init__(self):
        self._repository = DocumentRepository()
        self._insight_kernel = InsightKernel()
        self._summary_kernel = SummaryKernel()
        self.vector_store = VectorStoreFactory.get_instance()
        self.embedding_service = EmbeddingService()

    def create_document(self, data: CreateDocumentRequest) -> Document:
        """
        create new document
        Args:
            data (CreateDocumentRequest): create doc request
        Returns:
            Document: create doc object
        """
        doc = Document(
            name=data.name,
            title=data.title,
            mime_type=data.mime_type,
            user_description=data.user_description,
            url=str(data.url) if data.url else None,
            document_size=data.document_size,
            extract_status=data.extract_status,
            embedding_status=ProcessStatus.INITIALIZED,
            raw_content=data.raw_content,
        )
        return self._repository.create(doc)

    def _invalidate_l1_summaries(self) -> None:
        # Stale L1 is skipped by retrieval and training; status bios stay until L1 regenerates.
        from lpm_kernel.models.l1 import L1Version
        with DatabaseSession.session() as session:
            session.query(L1Version).update({L1Version.status: "stale"})
            session.commit()

    def refresh_document_index(self, document_id: int, raw_content: Optional[str] = None):
        """Refresh factual memory without analysis, GraphRAG or LoRA retraining.

        A failed embedding leaves the document retryable, and removes outdated
        vectors first so changed/deleted facts cannot keep being retrieved.
        """
        from lpm_kernel.file_data.chunker import DocumentChunker
        from lpm_kernel.file_data.models import ChunkModel
        from lpm_kernel.configs.config import Config
        document = self._repository.find_one(document_id)
        if not document:
            raise ValueError("Document not found")
        content = document.raw_content if raw_content is None else raw_content
        if not content or not content.strip():
            raise ValueError("Document content cannot be empty")
        config = Config.from_env()
        chunks = DocumentChunker(
            chunk_size=int(config.get("DOCUMENT_CHUNK_SIZE") or 1000),
            overlap=int(config.get("DOCUMENT_CHUNK_OVERLAP") or 200),
        ).split(content)
        self.embedding_service.delete_document_index(document_id)
        self._invalidate_l1_summaries()
        with self._repository._db.session() as session:
            model = session.get(Document, document_id)
            model.raw_content = content
            model.embedding_status = ProcessStatus.INITIALIZED
            if raw_content is not None:
                model.insight = None
                model.summary = None
                model.analyze_status = ProcessStatus.INITIALIZED
            session.query(ChunkModel).filter(ChunkModel.document_id == document_id).delete()
            from sqlalchemy import func
            # Existing SQLite databases may have BIGINT primary keys, which do
            # not gain rowid autoincrement merely by updating the ORM type.
            next_id = (session.query(func.max(ChunkModel.id)).scalar() or 0) + 1
            session.add_all([ChunkModel(id=next_id + index, document_id=document_id,
                                        content=chunk.content, tags=chunk.tags, topic=chunk.topic)
                             for index, chunk in enumerate(chunks)])
            session.commit()
        try:
            embedding = self.process_document_embedding(document_id)
            embedded_chunks = self.generate_document_chunk_embeddings(document_id)
            if embedding is None or not embedded_chunks or not all(c.has_embedding for c in embedded_chunks):
                raise RuntimeError("Memory indexing failed; retry the document indexing operation")
            return {"document_id": document_id, "total_chunks": len(embedded_chunks)}
        except Exception:
            self.embedding_service.delete_document_index(document_id)
            self._repository.update_embedding_status(document_id, ProcessStatus.FAILED)
            raise

    def list_documents(self) -> List[Document]:
        """
        get all doc list
        Returns:
            List[Document]: doc object list
        """
        return self._repository.list()

    def scan_directory(
        self, directory_path: str, recursive: bool = False
    ) -> List[DocumentDTO]:
        """
        scan and process files
        Args:
            directory_path (str): dir to scan
            recursive (bool, optional): if recursive scan. Defaults to False.
        Returns:
            List[Document]: processed doc object list
        Raises:
            FileProcessingError: when dir not exist or failed
        """

        path = Path(directory_path)
        if not path.is_dir():
            raise FileProcessingError(f"{directory_path} is not a directory")

        documents_dtos: List[DocumentDTO] = []
        pattern = "**/*" if recursive else "*"

        # list all files
        files = list(path.glob(pattern))
        logger.info(f"Found files: {files}")

        for file_path in files:
            if file_path.is_file():
                try:
                    logger.info(f"Processing file: {file_path}")
                    doc = ProcessorFactory.auto_detect_and_process(str(file_path))

                    # create CreateDocumentRequest obj to database
                    request = CreateDocumentRequest(
                        name=doc.name,
                        title=doc.name,
                        mime_type=doc.mime_type,
                        user_description="Auto scanned document",
                        document_size=doc.document_size,
                        url=str(file_path.absolute()),
                        raw_content=doc.raw_content,
                        extract_status=doc.extract_status,
                        embedding_status=ProcessStatus.INITIALIZED,
                    )
                    saved_doc = self.create_document(request)

                    documents_dtos.append(saved_doc.to_dto())
                    logger.info(f"Successfully processed and saved: {file_path}")

                except Exception:
                    # add detailed error log
                    logger.exception(
                        f"Error processing file {file_path}"
                    )
                    continue

        logger.info(f"Total documents processed and saved: {len(documents_dtos)}")
        return documents_dtos

    def _analyze_document(self, doc: DocumentDTO) -> DocumentDTO:
        """
        analyze one file
        Args:
            doc (Document): doc to analyze
        Returns:
            Document: updated doc
        Raises:
            Exception: error occurred
        """
        try:
            # generate insight
            insight_result = self._insight_kernel.analyze(doc)

            # generate summary
            summary_result = self._summary_kernel.analyze(
                doc, insight_result["insight"]
            )

            # update database
            updated_doc = self._repository.update_document_analysis(
                doc.id, insight_result, summary_result
            )

            return updated_doc

        except Exception as e:
            logger.error(f"Document {doc.id} analysis failed: {str(e)}", exc_info=True)
            # update status as failed
            self._update_analyze_status_failed(doc.id)
            raise

    def analyze_document(self, document_id: int) -> DocumentDTO:
        """
        Analyze a single document by ID
        
        Args:
            document_id (int): ID of document to analyze
            
        Returns:
            DocumentDTO: The analyzed document
            
        Raises:
            ValueError: If document not found
            Exception: If analysis fails
        """
        try:
            # Get document
            document = self._repository.find_one(document_id)
            if not document:
                raise ValueError(f"Document not found with id: {document_id}")
                
            # Perform analysis
            return self._analyze_document(document)
            
        except ValueError as e:
            logger.error(f"Document {document_id} not found: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error analyzing document {document_id}: {str(e)}", exc_info=True)
            self._update_analyze_status_failed(document_id)
            raise

    def _update_analyze_status_failed(self, doc_id: int) -> None:
        """update status as failed"""
        try:
            with self._repository._db.session() as session:
                document = session.get(self._repository.model, doc_id)
                if document:
                    document.analyze_status = ProcessStatus.FAILED
                    session.commit()
                    logger.debug(f"Updated analyze status for document {doc_id} to FAILED")
                else:
                    logger.warning(f"Document not found with id: {doc_id}")
        except Exception as e:
            logger.error(f"Error updating document analyze status: {str(e)}")

    def check_all_documents_embeding_status(self) -> bool:
        """
        Check if there are any documents that need embedding
        Returns:
            bool: True if there are documents that need embedding, False otherwise
        """
        try:
            unembedding_docs = self._repository.find_unembedding()
            return len(unembedding_docs) > 0
        except Exception as e:
            logger.error(f"Error checking documents embedding status: {str(e)}", exc_info=True)
            raise

    def analyze_all_documents(self) -> List[DocumentDTO]:
        """
        analyze all unanalyzed documents
        Returns:
            List[DocumentDTO]: finished doc list
        Raises:
            Exception: error occurred
        """
        try:
            # get all unanalyzed documents
            unanalyzed_docs = self._repository.find_unanalyzed()

            analyzed_docs = []
            success_count = 0
            error_count = 0

            for index, doc in enumerate(unanalyzed_docs, 1):
                try:
                    analyzed_doc = self._analyze_document(doc)
                    analyzed_docs.append(analyzed_doc)
                    success_count += 1
                except Exception as e:
                    error_count += 1
                    logger.error(f"Document {doc.id} processing failed: {str(e)}")
                    continue

            return analyzed_docs

        except Exception as e:
            logger.error(f"Error occurred during batch analysis: {str(e)}", exc_info=True)
            raise

    def get_document_l0(self, document_id: int) -> Dict:
        """
        get chunks and embeds
        Args:
            document_id (int): doc ID
        Returns:
            Dict: format:
                {
                    "document_id": int,
                    "chunks": List[Dict],
                    "total_chunks": int
                }
        Raises:
            FileProcessingError: doc not existed
        """
        try:
            # get doc
            document = self._repository.find_one(document_id)
            if not document:
                raise FileProcessingError(f"Document not found: {document_id}")

            # get doc chunks
            chunks = self.get_document_chunks(document_id)
            if not chunks:
                return {"document_id": document_id, "chunks": [], "total_chunks": 0}

            # get doc embeddings
            all_chunk_embeddings = self.get_chunk_embeddings_by_document_id(document_id)

            # get L0 data
            l0_data = {
                "document_id": document_id,
                "chunks": [
                    {
                        "id": chunk.id,
                        "content": chunk.content,
                        "has_embedding": chunk.has_embedding,
                        "embedding": all_chunk_embeddings.get(chunk.id),
                        "tags": chunk.tags,
                        "topic": chunk.topic,
                    }
                    for chunk in chunks
                ],
                "total_chunks": len(chunks),
            }

            return l0_data

        except FileProcessingError as e:
            raise e
        except Exception as e:
            logger.error(f"Error getting L0 data for document {document_id}: {str(e)}")
            raise FileProcessingError(f"Failed to get L0 data: {str(e)}")

    def get_document_chunks(self, document_id: int) -> List[ChunkDTO]:
        """
        get chunks result
        Args:
            document_id (int): doc ID
        Returns:
            List[ChunkDTO]: doc chunks list，each ChunkDTO include embedding info
        """
        try:
            document = self._repository.find_one(document_id=document_id)
            if not document:
                logger.info(f"Document not found with id: {document_id}")
                return []

            chunks = self._repository.find_chunks(document_id=document_id)
            logger.info(f"Found {len(chunks)} chunks for document {document_id}")

            for chunk in chunks:
                chunk.length = len(chunk.content) if chunk.content else 0
                if chunk.has_embedding:
                    chunk.embedding = (
                        self.embedding_service.get_chunk_embedding_by_chunk_id(chunk.id)
                    )

            return chunks

        except Exception as e:
            logger.error(f"Error getting chunks for document {document_id}: {str(e)}")
            return []

    # def save_chunk(self, chunk: Chunk) -> None:
    #     """
    #     Args:
    #         chunk (Chunk): chunk obj
    #     Raises:
    #         Exception: error occurred
    #     """
    #     try:
    #         # create ChunkModel instance
    #         chunk_model = ChunkModel(
    #             document_id=chunk.document_id,
    #             content=chunk.content,
    #             tags=chunk.tags,
    #             topic=chunk.topic,
    #         )
    #         # save to db
    #         self._repository.save_chunk(chunk_model)
    #         logger.debug(f"Saved chunk for document {chunk.document_id}")
    #     except Exception as e:
    #         logger.error(f"Error saving chunk: {str(e)}")
    #         raise

    def list_documents_with_l0(self) -> List[Dict]:
        """
        get all docs' L0 data
        Returns:
            List[Dict]: list of dict of docs with L0 data
        """
        # 1. get all basic data
        documents = self.list_documents()
        logger.info(f"list_documents len: {len(documents)}")

        # 2. each doc L0
        documents_with_l0 = []
        for doc in documents:
            doc_dict = doc.to_dict()
            try:
                l0_data = self.get_document_l0(doc.id)
                doc_dict["l0_data"] = l0_data
                logger.info(f"success getting L0 data for document {doc.id} success")
            except Exception as e:
                logger.error(f"Error getting L0 data for document {doc.id}: {str(e)}")
                doc_dict["l0_data"] = None
            documents_with_l0.append(doc_dict)

        return documents_with_l0

    def get_document_by_id(self, document_id: int) -> Optional[Document]:
        """
        get doc by ID
        Args:
            document_id (int): doc ID
        Returns:
            Optional[Document]: doc object, None if not found
        """
        try:
            return self._repository.find_one(document_id)
        except Exception as e:
            logger.error(f"Error getting document by id {document_id}: {str(e)}")
            return None

    def generate_document_chunk_embeddings(self, document_id: int) -> List[ChunkDTO]:
        """
        handle chunks and embeddings
        Args:
            document_id (int): ID
        Returns:
            List[ChunkDTO]: chunks list
        Raises:
            Exception: error occurred
        """
        try:
            chunks_dtos = self._repository.find_chunks(document_id)
            if not chunks_dtos:
                logger.info(f"No chunks found for document {document_id}")
                return []

            # handle embeddings
            processed_chunks = self.embedding_service.generate_chunk_embeddings(
                chunks_dtos
            )

            # update state in db
            for chunk_dto in processed_chunks:
                if chunk_dto.has_embedding:
                    self._repository.update_chunk_embedding_status(chunk_dto.id, True)

            return processed_chunks

        except Exception as e:
            logger.error(f"Error processing chunk embeddings: {str(e)}")
            raise

    def get_chunk_embeddings_by_document_id(
        self, document_id: int
    ) -> Dict[int, List[float]]:
        """
        get chunks embeddings
        Args:
            document_id (int): doc ID
        Returns:
            Dict[int, List[float]]: chunk_id to embedding mapping
        Raises:
            Exception: error occurred
        """
        try:
            # get all chunks ID
            chunks = self._repository.find_chunks(document_id)
            chunk_ids = [str(chunk.id) for chunk in chunks]

            self.embedding_service._ensure_current_model()
            # get embeddings from ChromaDB
            embeddings = {}
            if chunk_ids:
                results = self.embedding_service.chunk_collection.get(
                    ids=chunk_ids, include=["embeddings", "documents"]
                )

                # transfer chunk_id -> embedding
                for i, chunk_id in enumerate(results["ids"]):
                    embeddings[int(chunk_id)] = results["embeddings"][i]

            return embeddings

        except Exception as e:
            logger.error(
                f"Error getting chunk embeddings for document {document_id}: {str(e)}"
            )
            raise

    def process_document_embedding(self, document_id: int) -> List[float]:
        """
        handle doc level embedding
        Args:
            document_id (int): doc ID
        Returns:
            List[float]: doc embedding
        Raises:
            ValueError: doc not exist
            Exception: error occurred
        """
        try:
            document = self._repository.find_one(document_id)
            if not document:
                raise ValueError(f"Document not found with id: {document_id}")

            if not document.raw_content:
                logger.warning(
                    f"Document {document_id} has no content to process embedding"
                )
                self._repository.update_embedding_status(
                    document_id, ProcessStatus.FAILED
                )
                return None

            # gen doc embedding
            embedding = self.embedding_service.generate_document_embedding(document)
            if embedding is not None:
                self._repository.update_embedding_status(
                    document_id, ProcessStatus.SUCCESS
                )
            else:
                self._repository.update_embedding_status(
                    document_id, ProcessStatus.FAILED
                )

            return embedding

        except Exception as e:
            logger.error(f"Error processing document embedding: {str(e)}")
            self._repository.update_embedding_status(document_id, ProcessStatus.FAILED)
            raise

    def get_document_embedding(self, document_id: int) -> Optional[List[float]]:
        """
        get doc embedding
        Args:
            document_id (int): doc ID
        Returns:
            Optional[List[float]]: doc embedding
        Raises:
            Exception: error occurred
        """
        try:
            self.embedding_service._ensure_current_model()
            results = self.embedding_service.document_collection.get(
                ids=[str(document_id)], include=["embeddings"]
            )

            if results and results.get("embeddings") is not None and len(results["embeddings"]) > 0:
                return results["embeddings"][0]
            return None

        except Exception as e:
            logger.error(f"Error getting document embedding: {str(e)}")
            raise

    def delete_file_by_name(self, filename: str) -> bool:
        """
        Args:
            filename (str): name to delete
            
        Returns:
            bool: if success
            
        Raises:
            Exception: error occurred
        """
        logger.info(f"Starting to delete file: {filename}")
        
        try:
            # 1. search memories
            db = DatabaseSession()
            memory = None
            document_id = None
            
            with db._session_factory() as session:
                query = select(Memory).where(Memory.name == filename)
                result = session.execute(query)
                memory = result.scalar_one_or_none()
                
                if not memory:
                    logger.warning(f"File record not found: {filename}")
                    return False
                
                # get related document_id
                document_id = memory.document_id
                
                # get filepath
                file_path = memory.path
                
                # Clear every model index before deleting the database record.
                if document_id:
                    self.embedding_service.delete_document_index(int(document_id))
                    self._invalidate_l1_summaries()
                # 2. delete memory
                session.delete(memory)
                session.commit()
                logger.info(f"Deleted record from memories table: {filename}")
            
            # if no related document, only delete physical file
            if not document_id:
                # delete physical file
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Deleted physical file: {file_path}")
                return True
            
            # 3. get doc obj
            document = self._repository.get_by_id(document_id)
            if not document:
                logger.warning(f"Corresponding document record not found, ID: {document_id}")
                # if no document record, delete physical file
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Deleted physical file: {file_path}")
                return True
            
            # 7. delete all chunks embedding from ChromaDB
            with db._session_factory() as session:
                from lpm_kernel.file_data.models import ChunkModel
                session.query(ChunkModel).filter(
                    ChunkModel.document_id == document_id
                ).delete()
                session.commit()
                logger.info("Deleted all related chunks")
                
                # delete doc record
                doc_entity = session.get(Document, document_id)
                if doc_entity:
                    session.delete(doc_entity)
                    session.commit()
                    logger.info(f"Deleted document record from database, ID: {document_id}")
            
            # 8. delete physical file
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted physical file: {file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting file: {str(e)}", exc_info=True)
            raise

    def fix_missing_document_analysis(self) -> int:
        """Fix documents with missing insights or summaries
        
        Returns:
            int: Number of documents fixed
        """
        try:
            # Find all documents that have analysis issues
            docs = self._repository.list()
            fixed_count = 0
            
            for doc in docs:
                needs_fixing = False
                
                # Check if document needs analysis
                if not doc.analyze_status or doc.analyze_status != ProcessStatus.SUCCESS:
                    needs_fixing = True
                    logger.info(f"Document {doc.id} needs analysis (status: {doc.analyze_status})")
                
                # Check if document has missing insights or summaries
                elif not doc.insight or not doc.summary:
                    needs_fixing = True
                    logger.info(f"Document {doc.id} has missing insight or summary")
                
                # Process documents that need fixing
                if needs_fixing:
                    try:
                        # Process document analysis
                        self.analyze_document(doc.id)
                        fixed_count += 1
                        logger.info(f"Fixed document {doc.id} analysis")
                    except Exception as e:
                        logger.error(f"Error fixing document {doc.id} analysis: {str(e)}")
                
            logger.info(f"Fixed {fixed_count} documents with missing analysis")
            return fixed_count
            
        except Exception as e:
            logger.error(f"Error in fix_missing_document_analysis: {str(e)}")
            raise FileProcessingError(f"Failed to fix document analysis: {str(e)}")

    def verify_document_embeddings(self, verbose=True) -> Dict:
        """
        Verify all document embeddings and return statistics
        
        Args:
            verbose (bool): Whether to log detailed information
            
        Returns:
            Dict: Statistics about document embeddings
        """
        try:
            docs = self._repository.list()
            results = {
                "total_documents": len(docs),
                "documents_with_embedding": 0,
                "documents_without_embedding": 0,
                "documents_with_content": 0,
                "documents_without_content": 0,
                "documents_with_summary": 0,
                "documents_without_summary": 0,
                "documents_with_insight": 0,
                "documents_without_insight": 0,
                "documents_needing_repair": 0,
            }
            
            documents_needing_repair = []
            
            for doc in docs:
                # Check if document has content
                if doc.raw_content:
                    results["documents_with_content"] += 1
                else:
                    results["documents_without_content"] += 1
                    
                # Check if document has summary
                if doc.summary:
                    results["documents_with_summary"] += 1
                else:
                    results["documents_without_summary"] += 1
                    
                # Check if document has insight
                if doc.insight:
                    results["documents_with_insight"] += 1
                else:
                    results["documents_without_insight"] += 1
                
                # Check if embeddings exist in ChromaDB
                embedding = self.get_document_embedding(doc.id)
                if embedding is not None:
                    results["documents_with_embedding"] += 1
                    if verbose:
                        logger.info(f"Document {doc.id}: '{doc.name}' has embedding of dimension {len(embedding)}")
                else:
                    results["documents_without_embedding"] += 1
                    if verbose:
                        logger.warning(f"Document {doc.id}: '{doc.name}' missing embedding")
                    
                # Check if document needs repair (has content but missing embedding or analysis)
                if doc.raw_content and (embedding is None or not doc.summary or not doc.insight):
                    documents_needing_repair.append(doc.id)
                    results["documents_needing_repair"] += 1
                    
            # Log statistics
            logger.info(f"Document embedding verification results: {results}")
            if documents_needing_repair and verbose:
                logger.info(f"Documents needing repair: {documents_needing_repair}")
                
            return results
            
        except Exception as e:
            logger.error(f"Error verifying document embeddings: {str(e)}", exc_info=True)
            raise


# create service
document_service = DocumentService()

# use elsewhere by:
# from lpm_kernel.file_data.service import document_service
