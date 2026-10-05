"""
service about knowledge retrieve
"""
import os
import logging
from typing import List, Optional, Tuple
from lpm_kernel.file_data.embedding_service import EmbeddingService, ChunkDTO
from lpm_kernel.kernel.l1.l1_manager import get_latest_global_bio

logger = logging.getLogger(__name__)

# Cosine scales differ per embedding model: with bge-m3, answerable questions
# score about 0.52-0.68 and unanswerable ones 0.34-0.47.
MODEL_SIMILARITY_THRESHOLDS = {"bge-m3": 0.5}
DEFAULT_SIMILARITY_THRESHOLD = 0.7


def default_similarity_threshold(model_name: Optional[str]) -> float:
    name = (model_name or "").lower()
    return next((threshold for key, threshold in MODEL_SIMILARITY_THRESHOLDS.items() if key in name),
                DEFAULT_SIMILARITY_THRESHOLD)


def resolve_similarity_threshold(embedding_service, env_name: str) -> float:
    """Env override first, else the default for the currently configured embedding model."""
    override = os.getenv(env_name)
    if override:
        return float(override)
    config_service = getattr(getattr(embedding_service, "llm_client", None), "user_llm_config_service", None)
    configuration = config_service.get_available_llm() if config_service else None
    return default_similarity_threshold(getattr(configuration, "embedding_model_name", None))


class L0KnowledgeRetriever:
    """L0 knowledge retriever"""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        similarity_threshold: Optional[float] = None,
        max_chunks: int = 3,
    ):
        """
        init L0 knowledge retriever

        Args:
            embedding_service: Embedding service instance
            similarity_threshold: only return contents whose similarity bigger than this value;
                None resolves it per embedding model at retrieve time
            max_chunks: the maximum number of return chunks
        """
        self.embedding_service = embedding_service
        self.similarity_threshold = similarity_threshold
        self.max_chunks = max_chunks

    def retrieve(self, query: str) -> str:
        """
        retrieve L0 knowledge

        Args:
            query: query content

        Returns:
            str: structured knowledge content, or empty string if no relevant knowledge found
        """
        try:
            # search related chunks
            similar_chunks: List[
                Tuple[ChunkDTO, float]
            ] = self.embedding_service.search_similar_chunks(
                query=query, limit=self.max_chunks
            )

            # filter out low similarity chunks
            if not similar_chunks:
                return ""

            threshold = self.similarity_threshold
            if threshold is None:
                threshold = resolve_similarity_threshold(self.embedding_service, "L0_SIMILARITY_THRESHOLD")
            knowledge_parts = []
            for chunk, similarity in similar_chunks:
                if similarity >= threshold:
                    knowledge_parts.append(f"[document:{chunk.document_id} chunk:{chunk.id}]\n{chunk.content}")

            if not knowledge_parts:
                return ""

            # merge multiple knowledge parts into one
            return "\n\n".join(knowledge_parts)

        except Exception as e:
            logger.error(f"L0 knowledge retrieval failed: {str(e)}")
            return ""


class L1KnowledgeRetriever:
    """L1 knowledge retriever"""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        similarity_threshold: Optional[float] = None,
        max_shades: int = 3,
    ):
        """
        init L1 knowledge retriever

        Args:
            embedding_service: Embedding service instance
            similarity_threshold: only return contents whose similarity bigger than this value;
                None resolves it per embedding model at retrieve time
            max_shades: the maximum number of return shades
        """
        self.embedding_service = embedding_service
        self.similarity_threshold = similarity_threshold
        self.max_shades = max_shades
        self._shade_cache_key = None
        self._shade_embeddings = []

    def retrieve(self, query: str) -> str:
        """
        search related L1 shades

        Args:
            query: query content

        Returns:
            str: structured knowledge content, or empty string if no relevant knowledge found
        """
        try:
            # get global bio shades
            global_bio = get_latest_global_bio()
            if not global_bio or not global_bio.shades or global_bio.status == "stale":
                logger.info("Global Bio not found or Shades is empty")
                return ""

            # get query embedding
            query_embedding = self.embedding_service.get_embedding(query)
            if query_embedding is None or len(query_embedding) == 0:
                logger.error("Failed to get embedding for query text")
                return ""

            # Cache by full shade content so edits invalidate the vectors.
            key = (getattr(self.embedding_service, "embedding_space", None), tuple((shade.get("id"), shade.get("title", ""),
                         shade.get("description", ""), shade.get("content", ""))
                        for shade in global_bio.shades))
            if key != self._shade_cache_key:
                texts = [f"{shade.get('title', '')} - {shade.get('description', '')}\n"
                         f"{shade.get('content', '')}" for shade in global_bio.shades]
                embeddings = self.embedding_service.llm_client.get_embedding(texts)
                if embeddings is None or len(embeddings) != len(texts):
                    raise ValueError("Incomplete shade embedding batch")
                self._shade_embeddings = list(zip(global_bio.shades, embeddings))
                self._shade_cache_key = key
            shade_embeddings = self._shade_embeddings

            if not shade_embeddings:
                logger.info("No available Shades embeddings found")
                return ""

            # calculate similarity and sort
            threshold = self.similarity_threshold
            if threshold is None:
                threshold = resolve_similarity_threshold(self.embedding_service, "L1_SIMILARITY_THRESHOLD")
            similar_shades = []
            for shade, embedding in shade_embeddings:
                similarity = self.embedding_service.calculate_similarity(
                    query_embedding, embedding
                )
                if similarity >= threshold:
                    similar_shades.append((shade, similarity))

            # sort according to similarity and limit the number of returned shades
            similar_shades.sort(key=lambda x: x[1], reverse=True)
            similar_shades = similar_shades[: self.max_shades]

            if not similar_shades:
                return ""

            # structured output
            shade_parts = []
            for shade, similarity in similar_shades:
                shade_text = f"Shade: {shade.get('title', '')}\n"
                shade_text += f"Description: {shade.get('description', '')}\n"
                shade_text += f"Content: {shade.get('content', '')}\n"
                shade_text += f"Similarity: {similarity:.2f}"
                shade_parts.append(shade_text)

            return "\n\n".join(shade_parts)

        except Exception as e:
            logger.error(f"L1 knowledge retrieval failed: {str(e)}")
            return ""


# create overall knowledge retriever instance; thresholds follow the embedding model
default_retriever = L0KnowledgeRetriever(
    embedding_service=EmbeddingService(),
    max_chunks=3,
)

default_l1_retriever = L1KnowledgeRetriever(
    embedding_service=EmbeddingService(), max_shades=3
)
