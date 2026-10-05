from collections import defaultdict
from typing import Any, Dict, List, Optional, Union
import copy
import itertools
import json
import math
import traceback

from lpm_kernel.common.performance import create_synthesis_client
from scipy.cluster.hierarchy import fcluster, linkage
import numpy as np

from lpm_kernel.L1.bio import Cluster, Memory, Note
from lpm_kernel.L1.prompt import (
    TOPICS_TEMPLATE_SYS,
    TOPICS_TEMPLATE_USR,
    SYS_COMB,
    USR_COMB,
)
from lpm_kernel.L1.utils import find_connected_components
from lpm_kernel.api.services.user_llm_config_service import UserLLMConfigService
from lpm_kernel.configs.logging import get_train_process_logger
logger = get_train_process_logger()


class TopicsGenerator:
    def __init__(self):
        """Initialize the TopicsGenerator with default parameters and configurations."""
        self.default_cophenetic_distance = 1.0
        self.default_outlier_cutoff_distance = 0.5
        self.default_cluster_merge_distance = 0.5
        self.topic_params = {
            "temperature": 0,
            "max_tokens": 1500,
            "top_p": 0,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "timeout": 180,
            "response_format": {"type": "json_object"},
        }
        self.user_llm_config_service = UserLLMConfigService()
        self.user_llm_config = self.user_llm_config_service.get_available_llm()
        if self.user_llm_config is None:
            self.client = None
            self.model_name = None
        else:
            self.client = create_synthesis_client(
                "l1_topics",
                api_key=self.user_llm_config.chat_api_key,
                base_url=self.user_llm_config.chat_endpoint,
            )
            self.model_name = self.user_llm_config.chat_model_name
        logger.info(f"user_llm_config: {self.user_llm_config}")
        self.threshold = 0.85
        self._top_p_adjusted = False  # Flag to track if top_p has been adjusted

    def _fix_top_p_param(self, error_message: str) -> bool:
        """Fixes the top_p parameter if an API error indicates it's invalid.
        
        Some LLM providers don't accept top_p=0 and require values in specific ranges.
        This function checks if the error is related to top_p and adjusts it to 0.001,
        which is close enough to 0 to maintain deterministic behavior while satisfying
        API requirements.
        
        Args:
            error_message: Error message from the API response.
            
        Returns:
            bool: True if top_p was adjusted, False otherwise.
        """
        if not self._top_p_adjusted and "top_p" in error_message.lower():
            logger.warning("Fixing top_p parameter from 0 to 0.001 to comply with model API requirements")
            self.topic_params["top_p"] = 0.001
            self._top_p_adjusted = True
            return True
        return False

    def _call_llm_with_retry(self, messages: List[Dict[str, str]], **kwargs) -> Any:
        """Calls the LLM API with automatic retry for parameter adjustments.
        
        This function handles making API calls to the language model while
        implementing automatic parameter fixes when errors occur. If the API
        rejects the call due to invalid top_p parameter, it will adjust the
        parameter value and retry the call once.
        
        Args:
            messages: List of messages for the API call.
            **kwargs: Additional parameters to pass to the API call.
            
        Returns:
            API response object from the language model.
            
        Raises:
            Exception: If the API call fails after all retries or for unrelated errors.
        """
        try:
            return self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **self.topic_params,
                **kwargs
            )
        except Exception as e:
            error_msg = str(e)
            logger.error(f"API Error: {error_msg}")
            
            # Try to fix top_p parameter if needed
            if hasattr(e, 'response') and hasattr(e.response, 'status_code') and e.response.status_code == 400:
                if self._fix_top_p_param(error_msg):
                    logger.info("Retrying LLM API call with adjusted top_p parameter")
                    return self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        **self.topic_params,
                        **kwargs
                    )
            
            # Re-raise the exception
            raise

    def __find_nearest_cluster(self, cluster_list: List[Cluster], memory: Memory) -> tuple:
        """
        Find the nearest cluster to a memory based on embedding distance.
        
        Args:
            cluster_list: List of clusters to search
            memory: Memory to find nearest cluster for
            
        Returns:
            A tuple containing (nearest_cluster, distance_to_cluster)
        """
        distances = [
            np.linalg.norm(memory.embedding - cluster.cluster_center)
            for cluster in cluster_list
        ]
        nearest_cluster_idx = np.argmin(distances)
        return cluster_list[nearest_cluster_idx], distances[nearest_cluster_idx]


    def __merge_closed_clusters(
        self, cluster_list: List[Cluster], cluster_merge_distance: float
    ) -> tuple:
        """
        Merge clusters that are close to each other based on the distance threshold.
        
        Args:
            cluster_list: List of clusters to check for merging
            cluster_merge_distance: Threshold distance for merging clusters
            
        Returns:
            A tuple containing (list_of_merged_cluster_ids, list_of_merged_clusters)
        """
        connected_clusters_list: List[List[Cluster]] = find_connected_components(
            cluster_list, cluster_merge_distance
        )
        connected_clusters_list = [cc for cc in connected_clusters_list if len(cc) > 1]
        merge_cluster_ids_list, merge_cluster_list = [], []
        for connected_clusters in connected_clusters_list:
            merge_cluster_ids = [cluster.cluster_id for cluster in connected_clusters]
            merge_cluster_ids_list.append(merge_cluster_ids)
            merge_cluster_list.append(self.__merge_clusters(connected_clusters))
        return merge_cluster_ids_list, merge_cluster_list


    def __merge_clusters(self, connected_clusters: List[Cluster]) -> Cluster:
        """
        Merge a list of connected clusters into a single cluster.
        
        Args:
            connected_clusters: List of clusters to merge
            
        Returns:
            A new merged cluster
        """
        new_cluster = Cluster(clusterId=connected_clusters[0].cluster_id, is_new=True)
        for cluster in connected_clusters:
            new_cluster.extend_memory_list(cluster.memory_list)
        new_cluster.merge_list = [
            cluster.cluster_id for cluster in connected_clusters if not cluster.is_new
        ]
        return new_cluster


    def _clusters_update_strategy(
        self,
        cluster_list: List[Cluster],
        outlier_memory_list: List[Memory],
        new_memory_list: List[Memory],
        cophenetic_distance: float,
        outlier_cutoff_distance: float,
        cluster_merge_distance: float,
    ) -> tuple:
        """
        Update existing clusters with new memories and handle outliers.
        
        Args:
            cluster_list: List of existing clusters
            outlier_memory_list: List of outlier memories from previous run
            new_memory_list: List of new memories to process
            cophenetic_distance: Distance threshold for hierarchical clustering
            outlier_cutoff_distance: Distance threshold to determine outliers
            cluster_merge_distance: Distance threshold for merging clusters
            
        Returns:
            A tuple containing (updated_clusters, new_outlier_memories)
        """
        updated_cluster_ids = set()

        for memory in new_memory_list:
            if memory.embedding is None:
                continue
            nearest_cluster, distance = self.__find_nearest_cluster(
                cluster_list, memory
            )
            if distance < outlier_cutoff_distance:
                nearest_cluster.add_memory(memory)
                updated_cluster_ids.add(nearest_cluster.cluster_id)
            else:
                outlier_memory_list.append(memory)

        merge_cluster_ids_list, merge_cluster_list = self.__merge_closed_clusters(
            cluster_list, cluster_merge_distance
        )
        updated_cluster_list = [
            cluster
            for cluster in cluster_list
            if cluster.cluster_id in list(updated_cluster_ids)
        ]
        updated_cluster_list = [
            cluster
            for cluster in updated_cluster_list
            if cluster.cluster_id not in list(itertools.chain(*merge_cluster_ids_list))
        ]

        # Initial calculation of size_threshold using updated_cluster_list
        size_threshold = math.sqrt(max([cluster.size for cluster in cluster_list]))

        # Merge updated_cluster_list and merge_cluster_list
        cluster_list = updated_cluster_list + merge_cluster_list

        # If the merged cluster_list is not empty, recalculate size_threshold
        if cluster_list:
            size_threshold = math.sqrt(max([cluster.size for cluster in cluster_list]))
        else:
            logger.info(
                "cluster_list after updated is empty, use size_threshold from raw cluster list"
            )

        if outlier_memory_list:
            (
                outlier_cluster_list,
                new_outlier_memory_list,
            ) = self._clusters_initial_strategy(
                outlier_memory_list, cophenetic_distance, size_threshold
            )
        else:
            outlier_cluster_list, new_outlier_memory_list = [], []

        return cluster_list + outlier_cluster_list, new_outlier_memory_list


    def _clusters_initial_strategy(
        self,
        memory_list: List[Memory],
        cophenetic_distance: float,
        size_threshold: int = None,
    ) -> tuple:
        """
        Initial clustering strategy for memories without existing clusters.
        
        Args:
            memory_list: List of memories to cluster
            cophenetic_distance: Distance threshold for hierarchical clustering
            size_threshold: Minimum size threshold for valid clusters
            
        Returns:
            A tuple containing (generated_clusters, outlier_memories)
        """
        for memory in memory_list:
            logger.info(f"memory embedding shape: {memory.embedding.shape}")
            logger.info(f"memory: {memory}")
        memory_embeddings = [memory.embedding for memory in memory_list]

        logger.info(f"memory_embeddings: {memory_embeddings}")

        if len(memory_embeddings) == 1:
            clusters = np.array([1])
        else:
            linked = linkage(memory_embeddings, method="ward")
            clusters = fcluster(linked, cophenetic_distance, criterion="distance")
        
        labels = clusters.tolist()

        cluster_dict = {}

        for memory, label in zip(memory_list, labels):
            if label not in cluster_dict:
                cluster_dict[label] = Cluster(clusterId=label, is_new=True)
            cluster_dict[label].add_memory(memory)

        cluster_list: List[Cluster] = self.__remove_immature_clusters(
            cluster_dict, size_threshold
        )
        # For initial strategy, we need remove some nodes near the cluster boundary, retaining the main components of the cluster.
        for cluster in cluster_list:
            cluster.prune_outliers_from_cluster()
        in_cluster_memory_list = [
            memory.memory_id
            for cluster in cluster_list
            for memory in cluster.memory_list
        ]
        outlier_memory_list = [
            memory
            for memory in memory_list
            if memory.memory_id not in in_cluster_memory_list
        ]

        logger.info(f"cluster_list: {cluster_list}")
        logger.info(f"outlier_memory_list: {outlier_memory_list}")

        return cluster_list, outlier_memory_list


    def __remove_immature_clusters(self, cluster_list: dict, size_threshold: int = None) -> List[Cluster]:
        """
        Remove clusters that are too small (immature).
        
        Args:
            cluster_list: Dictionary mapping cluster IDs to Cluster objects
            size_threshold: Size threshold below which clusters are considered immature
            
        Returns:
            List of clusters that meet the size threshold
        """
        if not size_threshold:
            max_cluster_size = max(cluster.size for cluster in cluster_list.values())
            size_threshold = math.sqrt(max_cluster_size)
        cluster_list = [
            cluster
            for _, cluster in cluster_list.items()
            if cluster.size >= size_threshold
        ]
        return cluster_list


    def generate_topics_for_shades(
        self,
        old_cluster_list,
        old_outlier_memory_list,
        new_memory_list,
        cophenetic_distance,
        outlier_cutoff_distance,
        cluster_merge_distance,
    ) -> dict:
        """
        Generate topic clusters for shades by updating existing clusters or creating new ones.
        
        Args:
            old_cluster_list: List of existing clusters
            old_outlier_memory_list: List of outlier memories from previous run
            new_memory_list: List of new memories to process
            cophenetic_distance: Distance threshold for hierarchical clustering
            outlier_cutoff_distance: Distance threshold to determine outliers
            cluster_merge_distance: Distance threshold for merging clusters
            
        Returns:
            A dictionary containing updated cluster list and outlier memory list
        """
        cophenetic_distance = cophenetic_distance or self.default_cophenetic_distance
        outlier_cutoff_distance = (
            outlier_cutoff_distance or self.default_outlier_cutoff_distance
        )
        cluster_merge_distance = (
            cluster_merge_distance or self.default_cluster_merge_distance
        )

        new_memory_list = [Memory(**memory) for memory in new_memory_list]
        new_memory_list = [
            memory for memory in new_memory_list if memory.embedding is not None
        ]

        old_cluster_list = [Cluster(**cluster) for cluster in old_cluster_list]
        old_outlier_memory_list = [
            Memory(**memory) for memory in old_outlier_memory_list
        ]

        if not old_cluster_list:
            # initial strategy
            cluster_list, outlier_memory_list = self._clusters_initial_strategy(
                new_memory_list, cophenetic_distance
            )
        else:
            # update strategy
            cluster_list, outlier_memory_list = self._clusters_update_strategy(
                old_cluster_list,
                old_outlier_memory_list,
                new_memory_list,
                cophenetic_distance,
                outlier_cutoff_distance,
                cluster_merge_distance,
            )

        logger.info(f"cluster_list num: {len(cluster_list)}")
        logger.info(
            f"in cluster memory num: {sum([len(cluster.memory_list) for cluster in cluster_list])}"
        )
        logger.info(f"outlier_memory_list num: {len(outlier_memory_list)}")

        return {
            "clusterList": [cluster.to_json() for cluster in cluster_list],
            "outlierMemoryList": [memory.to_json() for memory in outlier_memory_list],
        }


    def generate_topics(self, notes_list: List[Note]) -> dict:
        """
        Generate topics from a list of notes.
        
        Args:
            notes_list: List of Note objects to process
            
        Returns:
            A dictionary containing topic data
        """
        logger.info(f"notes_lst length: {len(notes_list)}")
        for i, note in enumerate(notes_list):
            logger.info(f"\nNote {i + 1}:")
            logger.info(f"  ID: {note.id}")
            logger.info(f"  Title: {note.title}")
            logger.info(f"  Content: {note.content[:200]}...")  # only showing first 200 characters
            logger.info(f"  Create Time: {note.create_time}")
            logger.info(f"  Memory Type: {note.memory_type}")
            logger.info(f"  Number of chunks: {len(note.chunks)}")
            for j, chunk in enumerate(note.chunks):
                logger.info(f"    Chunk {j + 1}:")
                logger.info(f"      ID: {chunk.id}")
                logger.info(f"      Document ID: {chunk.document_id}")
                logger.info(
                    f"      Content: {chunk.content[:100]}..."
                )  # only showing first 100 characters
                logger.info(f"      Has embedding: {chunk.embedding is not None}")
                if chunk.embedding is not None:
                    logger.info(f"      Embedding shape: {chunk.embedding.shape}")

        # notes clean pre-process
        tmpTopics = self._cold_start(notes_list)

        return tmpTopics


    def _cold_start(self, notes_list: List[Note]) -> dict:
        """
        Perform cold start clustering on a list of notes.
        
        Args:
            notes_list: List of Note objects to process
            
        Returns:
            A dictionary containing cluster data
        """
        embedding_matrix, clean_chunks, all_note_ids = self.__build_embedding_chunks(
            notes_list
        )
        logger.info(
            f"embedding_matrix shape: {len(embedding_matrix)}, clean_chunks length: {len(clean_chunks)}"
        )

        if len(embedding_matrix) == 0:
            logger.warning("No chunks found in the notes_lst")
            return None

        cluster_data = self.__cold_clusters(clean_chunks, embedding_matrix)
        return cluster_data


    def __cold_clusters(self, clean_chunks: List, embedding_matrix: List) -> dict:
        """
        Generate clusters from scratch using hierarchical clustering.
        
        Args:
            clean_chunks: List of cleaned chunks to process
            embedding_matrix: Matrix of embeddings for the chunks
            
        Returns:
            A dictionary containing cluster data
        """
        chunks_with_topics = self.__generate_topic_from_chunks(clean_chunks)
        if len(embedding_matrix) <= 1:
            # Directly form a single cluster with the current chunk
            chunk = chunks_with_topics[0]
            cluster_data = {}
            cluster_data[
                "0"
            ] = {  # Store the cluster data with a normalized cluster_id from 0 to len(cluster_data)
                "indices": [0],
                "docIds": [chunk.document_id],
                "contents": [chunk.content],
                "embedding": [chunk.embedding],
                "chunkIds": [chunk.id],
                "tags": chunk.tags,
                "topic": chunk.topic,
                "topicId": 0,
                "recTimes": 0,
            }
            return cluster_data

        Z = linkage(embedding_matrix, method="complete", metric="cosine")
        clusters = self.__collect_cluster_indices(Z, self.threshold)
        cluster_data = self.__gen_cluster_data(clusters, chunks_with_topics)

        return cluster_data


    def __collect_cluster_indices(self, Z: np.ndarray, threshold: float) -> dict:
        """
        Collect the leaf indices of each cluster from the linkage matrix.
        
        Args:
            Z: Linkage matrix from hierarchical clustering
            threshold: Distance threshold for forming clusters
            
        Returns:
            A dictionary mapping cluster IDs to lists of point indices in each cluster
        """
        clusters = defaultdict(list)
        n = Z.shape[0] + 1
        cluster_id = n
        for i, merge in enumerate(Z):
            left, right, dist, _ = merge
            if dist < threshold:
                if left < n:
                    clusters[cluster_id].append(int(left))
                else:
                    clusters[cluster_id].extend(clusters.pop(left))

                if right < n:
                    clusters[cluster_id].append(int(right))
                else:
                    clusters[cluster_id].extend(clusters.pop(right))

                cluster_id += 1

        # change the cluster_id to 0~len(clusters)
        new_cluster_id = 0
        new_clusters = {}
        for tmp_id, indices in clusters.items():
            new_clusters[new_cluster_id] = indices
            new_cluster_id += 1
        return new_clusters


    def __gen_cluster_data(self, clusters: dict, chunks_with_topics: List) -> dict:
        """
        Generate detailed cluster data from cluster indices and chunks.
        
        Args:
            clusters: Dictionary mapping cluster IDs to lists of point indices
            chunks_with_topics: List of chunks with topic information
            
        Returns:
            A dictionary containing detailed information for each cluster
        """
        cluster_data = {}
        docIds = [chunk.document_id for chunk in chunks_with_topics]
        contents = [chunk.content for chunk in chunks_with_topics]
        embeddings = [chunk.embedding for chunk in chunks_with_topics]
        tags = [chunk.tags for chunk in chunks_with_topics]
        topics = [chunk.topic for chunk in chunks_with_topics]
        chunkIds = [chunk.id for chunk in chunks_with_topics]
        topic_id = 0
        for cid, indices in clusters.items():
            c_tags = [tags[i] for i in indices]
            c_topics = [topics[i] for i in indices]

            # Assuming gen_cluster_topic is modified to handle lists
            new_tags, new_topic = self.__gen_cluster_topic(c_tags, c_topics)
            cluster_data[cid] = {
                "indices": indices,
                "docIds": [docIds[i] for i in indices],
                "contents": [contents[i] for i in indices],
                "embedding": [embeddings[i] for i in indices],
                "chunkIds": [chunkIds[i] for i in indices],
                "tags": new_tags,
                "topic": new_topic,
                "topicId": topic_id,
                "recTimes": 0,
            }
            topic_id += 1
        return cluster_data


    def __gen_cluster_topic(self, c_tags: List, c_topics: List) -> tuple:
        """
        Generate a combined topic and tags for a cluster.
        
        Args:
            c_tags: List of tags from chunks in the cluster
            c_topics: List of topics from chunks in the cluster
            
        Returns:
            A tuple containing (new_tags, new_topic)
        """
        messages = [
            {"role": "system", "content": SYS_COMB},
            {"role": "user", "content": USR_COMB.format(topics=c_topics, tags=c_tags)},
        ]
        res = self._call_llm_with_retry(messages)
        new_topic, new_tags = self.__parse_response(
            res.choices[0].message.content, "topic", "tags"
        )

        return new_tags, new_topic


    def __generate_topic_from_chunks(self, chunks: List) -> List:
        """
        Generate topics and keywords for each chunk.
        
        Args:
            chunks: List of chunks to generate topics for
            
        Returns:
            List of chunks with added topic and tags information
        """
        chunks = copy.deepcopy(chunks)
        max_retries = 3  # maximum number of retries

        for chunk in chunks:
            for attempt in range(max_retries):
                try:
                    tmp_msg = [
                        {
                            "role": "system",
                            "content": TOPICS_TEMPLATE_SYS,
                        },
                        {
                            "role": "user",
                            "content": TOPICS_TEMPLATE_USR.format(chunk=chunk.content),
                        },
                    ]
                    logger.info(f"Attempt {attempt + 1}/{max_retries}")
                    logger.info(
                        f"Request messages: {json.dumps(tmp_msg, ensure_ascii=False)}"
                    )

                    answer = self._call_llm_with_retry(tmp_msg)
                    content = answer.choices[0].message.content
                    logger.info(f"Generated content: {content}")

                    topic, tags = self.__parse_response(content, "topic", "tags")
                    chunk.topic = topic
                    chunk.tags = tags
                    break  # exit the retry loop after a successful attempt

                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt == max_retries - 1:  # last attempt failed
                        logger.error(
                            f"All attempts failed for chunk: {traceback.format_exc()}"
                        )
                        # use default values or remove the chunk
                        chunk.topic = "Unknown Topic"  # set default value
                        chunk.tags = ["unclassified"]  # set default value
                        # or: chunks.remove(chunk)  # remove the chunk

            return chunks


    def __parse_response(self, content: str, key1: str, key2: str) -> tuple:
        """
        Parse JSON response to extract specific values.
        
        Args:
            content: JSON string to parse
            key1: First key to extract (typically 'topic')
            key2: Second key to extract (typically 'tags')
            
        Returns:
            A tuple containing the values for the two keys
        """
        spl = key1 + '":'
        b = '{"' + spl + "".join(content.split(spl)[1:])
        c = b.split("}")[0] + "}"
        res_dict = json.loads(c)

        return res_dict[key1], res_dict[key2]


    def __build_embedding_chunks(self, notes_list: List[Note]) -> tuple:
        """
        Build embedding matrix and clean chunks from a list of notes.
        
        Args:
            notes_list: List of Note objects to process
            
        Returns:
            A tuple containing (embedding_matrix, clean_chunks, all_note_ids)
        """
        all_chunks = [chunk for note in notes_list for chunk in note.chunks]
        all_chunks = [chunk for chunk in all_chunks if chunk.embedding is not None]
        all_note_ids = [note.id for note in notes_list]
        clean_chunks = []
        clean_ids = []
        clean_notes_lst = []
        # use content chunk
        for note_id in all_note_ids:
            tmp_chunks_set = [
                chunk for chunk in all_chunks if chunk.document_id == note_id
            ]
            if len(tmp_chunks_set) == 0:
                continue
            elif len(tmp_chunks_set) == 1:
                clean_chunks.append(tmp_chunks_set[0])
                clean_ids.append(note_id)
                clean_notes_lst.append(
                    [note for note in notes_list if note.id == note_id][0]
                )
            else:
                clean_ids.append(note_id)
                clean_notes_lst.append(
                    [note for note in notes_list if note.id == note_id][0]
                )
                for chunk in tmp_chunks_set:
                    clean_chunks.append(chunk)

        # form the embedding matrix
        embedding_matrix = [clean_chunk.embedding for clean_chunk in clean_chunks]

        return embedding_matrix, clean_chunks, all_note_ids
