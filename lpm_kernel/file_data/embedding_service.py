from typing import List, Tuple
import chromadb
import os
from .dto.chunk_dto import ChunkDTO
from lpm_kernel.common.llm import LLMClient
from lpm_kernel.file_data.document_dto import DocumentDTO
from typing import Optional
from lpm_kernel.configs.logging import get_train_process_logger
logger = get_train_process_logger()


class EmbeddingService:
    def __init__(self):
        from lpm_kernel.file_data.chroma_utils import detect_embedding_model_dimension, embedding_space_key
        from lpm_kernel.api.services.user_llm_config_service import UserLLMConfigService
        
        chroma_path = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.llm_client = LLMClient()
        
        # Get embedding model dimension from user config
        try:
            user_llm_config_service = UserLLMConfigService()
            user_llm_config = user_llm_config_service.get_available_llm()
            
            if user_llm_config and user_llm_config.embedding_model_name:
                # Detect dimension based on model name
                self.dimension = detect_embedding_model_dimension(user_llm_config.embedding_model_name)
                logger.info(f"Detected embedding dimension: {self.dimension} for model: {user_llm_config.embedding_model_name}")
            else:
                # Default to OpenAI dimension if no config found
                self.dimension = 1536
                logger.info(f"No embedding model configured, using default dimension: {self.dimension}")
        except Exception as e:
            # Default to OpenAI dimension if error occurs
            self.dimension = 1536
            logger.error(f"Error detecting embedding dimension, using default: {self.dimension}. Error: {str(e)}", exc_info=True)

        # Keep old collections intact when switching models (even same dimension).
        model_name = (user_llm_config.embedding_model_name if 'user_llm_config' in locals()
                      and user_llm_config and user_llm_config.embedding_model_name
                      else "text-embedding-ada-002")
        configuration = user_llm_config if 'user_llm_config' in locals() else None
        model_key = embedding_space_key(model_name,
                                        getattr(configuration, "embedding_provider_type", "") or "",
                                        getattr(configuration, "embedding_endpoint", "") or "")
        metadata = {"hnsw:space": "cosine", "dimension": self.dimension,
                    "embedding_model": model_name, "embedding_space": model_key}
        def collection_for(name):
            try:
                legacy = self.client.get_collection(name=name)
                legacy_metadata = legacy.metadata or {}
                same_model = legacy_metadata.get("embedding_space") == model_key
                legacy_openai = (not legacy_metadata.get("embedding_model")
                                 and model_name == "text-embedding-ada-002"
                                 and (getattr(configuration, "embedding_endpoint", "") or "").rstrip("/")
                                 in {"", "https://api.openai.com/v1"})
                if (legacy_metadata.get("dimension") == self.dimension
                        and legacy_metadata.get("hnsw:space") == "cosine"
                        and (same_model or legacy_openai)):
                    return legacy
            except Exception as error:
                if not isinstance(error, ValueError) and type(error).__name__ not in {"NotFoundError", "InvalidCollectionException"}:
                    raise
            isolated_name = f"{name}_{model_key}_{self.dimension}"
            return self.client.get_or_create_collection(name=isolated_name, metadata=metadata)
        self.embedding_model_name = model_name
        self.embedding_space = model_key
        self.document_collection = collection_for("documents")
        self.chunk_collection = collection_for("document_chunks")

    def _ensure_current_model(self):
        # The configuration can change while singleton services keep running.
        config_service = getattr(self.llm_client, "user_llm_config_service", None)
        if config_service is None:
            return
        configuration = config_service.get_available_llm()
        model = (configuration.embedding_model_name if configuration else None) or "text-embedding-ada-002"
        from lpm_kernel.file_data.chroma_utils import embedding_space_key
        space = embedding_space_key(model, getattr(configuration, "embedding_provider_type", "") or "",
                                    getattr(configuration, "embedding_endpoint", "") or "")
        if space != self.embedding_space:
            self.__init__()

    def generate_document_embedding(self, document: DocumentDTO) -> List[float]:
        """Process document level embedding and store in ChromaDB"""
        try:
            self._ensure_current_model()
            if not document.raw_content:
                logger.warning(
                    f"Document {document.id} has no content to process embedding"
                )
                return None

            # get embedding
            logger.info(f"Generating embedding for document {document.id}")
            embeddings = self.llm_client.get_embedding([document.raw_content])

            if embeddings is None or len(embeddings) == 0:
                logger.error(f"Failed to get embedding for document {document.id}")
                return None

            embedding = embeddings[0]
            logger.info(f"Successfully got embedding for document {document.id}")

            # store to ChromaDB
            try:
                logger.info(f"Storing embedding for document {document.id} in ChromaDB")
                self.document_collection.upsert(
                    documents=[document.raw_content],
                    ids=[str(document.id)],
                    embeddings=[embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)],
                    metadatas=[
                        {
                            "title": document.title or document.name or "",
                            "mime_type": document.mime_type or "",
                            "create_time": document.create_time.isoformat()
                            if document.create_time
                            else "",
                            "document_size": document.document_size or 0,
                            "url": document.url or "",
                        }
                    ],
                )
                logger.info(f"Successfully stored embedding for document {document.id}")

                # verify embedding storage
                result = self.document_collection.get(
                    ids=[str(document.id)], include=["embeddings"]
                )
                if not result or (result.get("embeddings") is None or len(result["embeddings"]) == 0):
                    logger.error(
                        f"Failed to verify embedding storage for document {document.id}"
                    )
                    return None
                logger.info(f"Verified embedding storage for document {document.id}")

                return embedding

            except Exception as e:
                logger.error(f"Error storing document embedding in ChromaDB: {str(e)}", exc_info=True)
                return None

        except Exception as e:
            logger.error(f"Error processing document embedding: {str(e)}", exc_info=True)
            raise

    def generate_chunk_embeddings(self, chunks: List[ChunkDTO]) -> List[ChunkDTO]:
        """Process chunk level embeddings"""
        """
        Store in ChromaDB, the structure is as follows:
        documents=[c.content for c in unprocessed_chunks],
                    ids=[str(c.id) for c in unprocessed_chunks],
                    embeddings=embeddings.tolist() if hasattr(embeddings, "tolist") else [list(v) for v in embeddings],
                    metadatas=[
                        {
                            "document_id": str(c.document_id),
                            "topic": c.topic or "",
                            "tags": ",".join(c.tags) if c.tags else "",
                        }
                        for c in unprocessed_chunks
                    ],
        """
        try:
            self._ensure_current_model()
            unprocessed_chunks = []
            for chunk in chunks:
                stored = self.chunk_collection.get(ids=[str(chunk.id)], include=["documents"])
                if not stored.get("ids") or stored.get("documents", [None])[0] != chunk.content:
                    chunk.has_embedding = False
                    unprocessed_chunks.append(chunk)
            if not unprocessed_chunks:
                logger.info("No unprocessed chunks found")
                return chunks

            logger.info(f"Processing embeddings for {len(unprocessed_chunks)} chunks")

            contents = [c.content for c in unprocessed_chunks]
            logger.info("Getting embeddings for %d chunks", len(contents))
            embeddings = self.llm_client.get_embedding(contents)

            if embeddings is None or len(embeddings) == 0:
                logger.error("Failed to get embeddings from LLM service")
                return chunks

            logger.info(f"Successfully got {len(embeddings)} embeddings")

            try:
                logger.info("Adding embeddings to ChromaDB...")
                self.chunk_collection.upsert(
                    documents=[c.content for c in unprocessed_chunks],
                    ids=[str(c.id) for c in unprocessed_chunks],
                    embeddings=embeddings.tolist() if hasattr(embeddings, "tolist") else [list(v) for v in embeddings],
                    metadatas=[
                        {
                            "document_id": str(c.document_id),
                            "topic": c.topic or "",
                            "tags": ",".join(c.tags) if c.tags else "",
                        }
                        for c in unprocessed_chunks
                    ],
                )
                logger.info("Successfully added embeddings to ChromaDB")

                # verify embeddings storage
                for chunk in unprocessed_chunks:
                    result = self.chunk_collection.get(
                        ids=[str(chunk.id)], include=["embeddings"]
                    )
                    if result and result.get("embeddings") is not None and len(result["embeddings"]) > 0:
                        chunk.has_embedding = True
                        logger.info(f"Verified embedding for chunk {chunk.id}")
                    else:
                        logger.warning(
                            f"Failed to verify embedding for chunk {chunk.id}"
                        )
                        chunk.has_embedding = False

            except Exception as e:
                logger.error(f"Error storing embeddings in ChromaDB: {str(e)}", exc_info=True)
                for chunk in unprocessed_chunks:
                    chunk.has_embedding = False
                raise

            return chunks

        except Exception as e:
            logger.error(f"Error processing chunk embeddings: {str(e)}", exc_info=True)
            raise

    def get_chunk_embedding_by_chunk_id(self, chunk_id: int) -> Optional[List[float]]:
        """Get the corresponding embedding vector by chunk_id

        Args:
            chunk_id (int): chunk ID

        Returns:
            List[float]: embedding vector, return None if not found

        Raises:
            ValueError: when chunk_id is invalid
            Exception: other errors
        """
        try:
            if not isinstance(chunk_id, int) or chunk_id < 0:
                raise ValueError("Invalid chunk_id")

            self._ensure_current_model()
            # query from ChromaDB
            result = self.chunk_collection.get(
                ids=[str(chunk_id)], include=["embeddings"]
            )

            if not result or (result.get("embeddings") is None or len(result["embeddings"]) == 0):
                logger.warning(f"No embedding found for chunk {chunk_id}")
                return None

            return result["embeddings"][0]

        except Exception as e:
            logger.error(f"Error getting embedding for chunk {chunk_id}: {str(e)}")
            raise

    def get_document_embedding_by_document_id(
        self, document_id: int
    ) -> Optional[List[float]]:
        """Get the corresponding embedding vector by document_id

        Args:
            document_id (int): document ID

        Returns:
            List[float]: embedding vector, return None if not found

        Raises:
            ValueError: when document_id is invalid
            Exception: other errors
        """
        try:
            if not isinstance(document_id, int) or document_id < 0:
                raise ValueError("Invalid document_id")

            self._ensure_current_model()
            # query from ChromaDB
            result = self.document_collection.get(
                ids=[str(document_id)], include=["embeddings"]
            )

            if not result or (result.get("embeddings") is None or len(result["embeddings"]) == 0):
                logger.warning(f"No embedding found for document {document_id}")
                return None

            return result["embeddings"][0]

        except Exception as e:
            logger.error(
                f"Error getting embedding for document {document_id}: {str(e)}"
            )
            raise

    def delete_document_index(self, document_id: int) -> None:
        """Remove only this document from every model-specific collection."""
        for entry in self.client.list_collections():
            name = entry if isinstance(entry, str) else entry.name
            if name == "document_chunks" or name.startswith("document_chunks_"):
                self.client.get_collection(name).delete(where={"document_id": str(document_id)})
            elif name == "documents" or name.startswith("documents_"):
                self.client.get_collection(name).delete(ids=[str(document_id)])

    def get_embedding(self, text: str) -> Optional[List[float]]:
        self._ensure_current_model()
        embeddings = self.llm_client.get_embedding([text])
        if embeddings is None or len(embeddings) == 0:
            return None
        vector = embeddings[0]
        return vector.tolist() if hasattr(vector, "tolist") else list(vector)

    @staticmethod
    def calculate_similarity(first: List[float], second: List[float]) -> float:
        """Cosine similarity, shared with cosine Chroma collections."""
        import math
        if len(first) != len(second):
            raise ValueError("Embedding dimensions differ")
        denominator = math.sqrt(sum(x*x for x in first) * sum(x*x for x in second))
        return sum(x*y for x, y in zip(first, second)) / denominator if denominator else 0.0

    def search_similar_chunks(
        self, query: str, limit: int = 5
    ) -> List[Tuple[ChunkDTO, float]]:
        """Search similar chunks, return list of ChunkDTO objects and their similarity scores

        Args:
            query (str): query text
            limit (int, optional): return result limit. Defaults to 5.

        Returns:
            List[Tuple[ChunkDTO, float]]: return list of (ChunkDTO, similarity score), sorted by similarity score in descending order

        Raises:
            ValueError: when query parameters are invalid
            Exception: other errors
        """
        try:
            if not query or not query.strip():
                raise ValueError("Query string cannot be empty")

            if limit < 1:
                raise ValueError("Limit must be positive")

            self._ensure_current_model()
            # calculate query text embedding
            query_embedding = self.llm_client.get_embedding([query])
            if query_embedding is None or len(query_embedding) == 0:
                raise Exception("Failed to generate embedding for query")

            # query ChromaDB
            results = self.chunk_collection.query(
                query_embeddings=[query_embedding[0].tolist() if hasattr(query_embedding[0], "tolist") else list(query_embedding[0])],
                n_results=limit,
                include=["documents", "metadatas", "distances"],
            )

            if not results or not results.get("ids") or not results["ids"][0]:
                return []

            # convert results to ChunkDTO objects
            similar_chunks = []
            ids = results["ids"][0]
            metadatas = (results.get("metadatas") or [[]])[0]
            documents = (results.get("documents") or [[]])[0]
            distances = (results.get("distances") or [[]])[0]
            if not (len(ids) == len(metadatas) == len(documents) == len(distances)):
                raise ValueError("Misaligned Chroma query results")
            for chunk_id, metadata, content, distance in zip(ids, metadatas, documents, distances):
                if not metadata or "document_id" not in metadata or content is None or distance is None:
                    continue
                document_id = metadata["document_id"]
                topic = metadata.get("topic", "")
                tags = metadata.get("tags", "").split(",") if metadata.get("tags") else []

                # Cosine distances are 1-cosine; squared L2 is not comparable.
                metric = (self.chunk_collection.metadata or {}).get("hnsw:space", "l2")
                if metric != "cosine":
                    raise ValueError("Legacy non-cosine index must be rebuilt before retrieval")
                similarity_score = 1 - distance

                chunk = ChunkDTO(
                    id=int(chunk_id),
                    document_id=int(document_id),
                    content=content,
                    topic=topic,
                    tags=tags,
                    has_embedding=True,
                )

                similar_chunks.append((chunk, similarity_score))

            # sort by similarity score in descending order
            similar_chunks.sort(key=lambda x: x[1], reverse=True)

            return similar_chunks

        except ValueError as ve:
            logger.error(f"Invalid input parameters: {str(ve)}")
            raise
        except Exception as e:
            logger.error(f"Error searching similar chunks: {str(e)}")
            raise