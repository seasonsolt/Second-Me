from typing import Optional
import numpy as np
from lpm_kernel.api.dto.user_llm_config_dto import (
    UserLLMConfigDTO,
)
from lpm_kernel.configs.logging import get_train_process_logger
logger = get_train_process_logger()
import requests

# Providers cap total tokens per request (e.g. Cloudflare bge-m3: 60k), so a large
# document must not be sent as a single request.
EMBEDDING_BATCH_SIZE = 8

def openai_strategy(user_llm_config: Optional[UserLLMConfigDTO], chunked_texts):
    headers = {
        "Authorization": f"Bearer {user_llm_config.embedding_api_key}",
        "Content-Type": "application/json",
    }

    logger.info("Getting embeddings with model %s, total chunks: %d",
                user_llm_config.embedding_model_name, len(chunked_texts))

    embeddings = []
    for start in range(0, len(chunked_texts), EMBEDDING_BATCH_SIZE):
        data = {
            "input": chunked_texts[start:start + EMBEDDING_BATCH_SIZE],
            "model": user_llm_config.embedding_model_name,
        }
        try:
            response = requests.post(
                f"{user_llm_config.embedding_endpoint}/embeddings", headers=headers, json=data
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            body = e.response.text[:500] if e.response is not None else ""
            raise Exception(f"Failed to get embeddings: {str(e)} {body}") from e

        # Extract embedding vectors
        embeddings.extend(item["embedding"] for item in response.json()["data"])

    return np.array(embeddings)
