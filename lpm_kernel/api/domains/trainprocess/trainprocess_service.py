import os
import re
import time
import psutil
from typing import Optional, Dict
from lpm_kernel.L1.utils import save_true_topics
from lpm_kernel.L1.serializers import NotesStorage
from lpm_kernel.kernel.note_service import NoteService
from lpm_kernel.L2.l2_generator import L2Generator
from lpm_kernel.L2.utils import save_hf_model
from lpm_kernel.api.domains.loads.services import LoadService
from lpm_kernel.kernel.chunk_service import ChunkService
from lpm_kernel.kernel.l1.l1_manager import (
    extract_notes_from_documents,
    document_service,
    get_latest_status_bio,
    get_latest_global_bio,
)
from lpm_kernel.configs.config import Config
from lpm_kernel.file_data.chunker import DocumentChunker
from lpm_kernel.kernel.l1.l1_manager import generate_l1_from_l0
import threading
from lpm_kernel.api.domains.trainprocess.progress_enum import Status
from lpm_kernel.api.domains.trainprocess.process_step import ProcessStep
from lpm_kernel.api.domains.trainprocess.progress_holder import TrainProgressHolder
from lpm_kernel.api.domains.trainprocess.training_params_manager import TrainingParamsManager
from lpm_kernel.common.repository.database_session import DatabaseSession
from lpm_kernel.common.performance import PerformanceRun
from lpm_kernel.L2.data_pipeline.data_prep.synthesis_config import get_synthesis_workers
from lpm_kernel.api.domains.kernel.routes import store_l1_data
from lpm_kernel.api.domains.trainprocess.L1_exposure_manager import output_files, query_l1_version_data, read_file_content
import gc
import subprocess
import sys
import signal
import json
from lpm_kernel.L2.mlx_training.backend import resolve_training_backend, normalize_training_language
from lpm_kernel.configs.logging import get_train_process_logger, TRAIN_LOG_FILE
logger = get_train_process_logger()

class TrainProcessService:
    """Training process service (singleton pattern)"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, current_model_name: str):
        if current_model_name is None:
            raise ValueError("current_model_name cannot be None")
            
        if not self._initialized:
            # Generate a unique progress file name based on model name
            self.progress = TrainProgressHolder(current_model_name)
            self.model_name = current_model_name  # Set model name directly
            self._initialized = True
            
            # Initialize stop flag
            self.is_stopped = False
            self.current_step = None
            self.process = None
            self._process_lock = threading.Lock()
            self._training_done = threading.Event()
            
            # Initialize L2 data dictionary
            self.l2_data = {
                "notes": None,
                "basic_info": None,
                "data_output_base_dir": None,
                "topics_path": None,
                "entitys_path": None,
                "graph_path": None,
                "config_path": None
            }
            self.l2_data_prepared = False
        
        # Update model name and progress instance if model name changes
        if current_model_name != self.model_name:
            self.model_name = current_model_name
            # Create new progress instance with updated progress file name
            self.progress = TrainProgressHolder(current_model_name)
    
    @classmethod
    def get_instance(cls, current_model_name: str = None):
        """Get the current instance of TrainProcessService
        
        Args:
            current_model_name: Optional model name to update the instance with
            
        Returns:
            TrainProcessService: The singleton instance
        """
        if cls._instance is None:
            if current_model_name is None:
                logger.warning("current_model_name must be provided when creating a new instance")
                return None
            return cls(current_model_name)
        
        if current_model_name is not None:
            # Update the existing instance with new model name
            cls._instance.model_name = current_model_name
            cls._instance.progress = TrainProgressHolder(current_model_name)
            
        return cls._instance

    def list_documents(self):
        """List all documents"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.LIST_DOCUMENTS, Status.IN_PROGRESS)            
            # Directly call document service instead of API
            documents = document_service.list_documents()
            # Mark step as completed if we found documents
            self.progress.mark_step_status(ProcessStep.LIST_DOCUMENTS, Status.COMPLETED)
                
            return [doc.to_dict() for doc in documents]
        except Exception as e:
            logger.error(f"List documents failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.LIST_DOCUMENTS, Status.FAILED)
            return []

    def generate_document_embeddings(self) -> bool:
        """Process embeddings for all documents"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.GENERATE_DOCUMENT_EMBEDDINGS, Status.IN_PROGRESS)
            documents = self.list_documents() 
            for doc in documents:
                doc_id = doc.get("id")

                # Directly call document service instead of API
                embedding = document_service.process_document_embedding(doc_id)
                if embedding is None:
                    logger.error(
                        f"Generate document embeddings failed for doc_id: {doc_id}"
                    )
                    self.progress.mark_step_status(ProcessStep.GENERATE_DOCUMENT_EMBEDDINGS, Status.FAILED)
                    return False
                self.progress.mark_step_status(ProcessStep.GENERATE_DOCUMENT_EMBEDDINGS, Status.COMPLETED)
                logger.info(f"Successfully generated embedding for document {doc_id}") 
            return True
        except Exception as e:
            logger.error(f"Generate document embeddings failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.GENERATE_DOCUMENT_EMBEDDINGS, Status.FAILED)
            return False

    def process_chunks(self) -> bool:
        """Process document chunks"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.CHUNK_DOCUMENT, Status.IN_PROGRESS)
            config = Config.from_env()
            chunker = DocumentChunker(
                chunk_size=int(config.get("DOCUMENT_CHUNK_SIZE")),
                overlap=int(config.get("DOCUMENT_CHUNK_OVERLAP")),
            )
            documents = document_service.list_documents()
            processed, failed = 0, 0

            chunk_service = ChunkService()
            for doc in documents:
                try:
                    if not doc.raw_content:
                        logger.warning(f"Document {doc.id} has no content, skipping...")
                        failed += 1
                        continue

                    # Uploaded memories may already have chunks indexed for live retrieval.
                    if document_service.get_document_chunks(doc.id):
                        processed += 1
                        continue
                    # Split only documents without chunks.
                    chunks = chunker.split(doc.raw_content)
                    for chunk in chunks:
                        chunk.document_id = doc.id
                        chunk_service.save_chunk(chunk)

                    processed += 1
                    logger.info(
                        f"Document {doc.id} processed: {len(chunks)} chunks created"
                    )
                except Exception as e:
                    logger.error(f"Failed to process document {doc.id}: {str(e)}")
                    failed += 1      
            self.progress.mark_step_status(ProcessStep.CHUNK_DOCUMENT, Status.COMPLETED)
            return True
        except Exception as e:
            logger.error(f"Process chunks failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.CHUNK_DOCUMENT, Status.FAILED)
            return False

    def chunk_embedding(self) -> bool:
        """Process embeddings for all document chunks"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.CHUNK_EMBEDDING, Status.IN_PROGRESS)
            documents = self.list_documents()
            for doc in documents:
                doc_id = doc.get("id")
                try:
                    # Directly call document service to generate chunk embeddings
                    processed_chunks = document_service.generate_document_chunk_embeddings(doc_id)
                    if not processed_chunks:
                        logger.warning(f"No chunks to process for document: {doc_id}")
                        continue
                except Exception as e:
                    logger.error(
                        f"Generate chunk embeddings failed for doc_id: {doc_id}: {str(e)}"
                    )
                    self.progress.mark_step_status(ProcessStep.CHUNK_EMBEDDING, Status.FAILED)
                    return False
            # All documents' chunks processed successfully
            self.progress.mark_step_status(ProcessStep.CHUNK_EMBEDDING, Status.COMPLETED)
            return True
        except Exception as e:
            logger.error(f"Generate chunk embeddings failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.CHUNK_EMBEDDING, Status.FAILED)
            return False

    def extract_dimensional_topics(self) -> bool:
        """Extract dimensional topics (L0)"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.EXTRACT_DIMENSIONAL_TOPICS, Status.IN_PROGRESS)
            logger.info("Starting dimensional topics extraction (L0)...")
            
            # Generate L0 - Call document_service to analyze all documents
            logger.info("Generating L0 data...")
            analyzed_docs = document_service.analyze_all_documents()
            logger.info(f"Successfully analyzed {len(analyzed_docs)} documents for L0")
            
            # Mark step as completed
            self.progress.mark_step_status(ProcessStep.EXTRACT_DIMENSIONAL_TOPICS, Status.COMPLETED)
            logger.info("Dimensional topics extraction (L0) completed successfully")
            return True

        except Exception as e:
            logger.error(f"Extract dimensional topics (L0) failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.EXTRACT_DIMENSIONAL_TOPICS, Status.FAILED)
            return False
            
    def generate_biography(self) -> bool:
        """Generate biography using L1 data"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.GENERATE_BIOGRAPHY, Status.IN_PROGRESS)
            logger.info("Starting biography generation...")

            # Generate L1 data and biography
            logger.info("Generating L1 data and biography...")
            l1_data = generate_l1_from_l0()
            logger.info("Successfully generated L1 data and biography")

            # Store L1 data
            with DatabaseSession.session() as session:
                store_l1_data(session, l1_data)

            # Mark step as completed
            self.progress.mark_step_status(ProcessStep.GENERATE_BIOGRAPHY, Status.COMPLETED)
            logger.info("Biography generation completed successfully")
            return True

        except Exception as e:
            logger.error(f"Biography generation failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.GENERATE_BIOGRAPHY, Status.FAILED)
            return False

    def model_download(self) -> bool:
        """Download model"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.MODEL_DOWNLOAD, Status.IN_PROGRESS)
            # Directly call save_hf_model function to download model
            logger.info(f"Starting model download: {self.model_name}")
            
            # Start monitoring the download progress in a separate thread
            monitor_thread = threading.Thread(target=self._monitor_model_download)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            # Start the actual download
            model_path = save_hf_model(self.model_name)
            
            if model_path and os.path.exists(model_path):
                logger.info(f"Model downloaded successfully to {model_path}")
                self.progress.mark_step_status(ProcessStep.MODEL_DOWNLOAD, Status.COMPLETED)
                return True
            else:
                logger.error(f"Model path does not exist after download: {model_path}")
                self.progress.mark_step_status(ProcessStep.MODEL_DOWNLOAD, Status.FAILED)
                return False

        except Exception as e:
            logger.error(f"Download model failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.MODEL_DOWNLOAD, Status.FAILED)
            return False

    def map_your_entity_network(self)->bool:
        """Map entity network using notes and basic info"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.MAP_ENTITY_NETWORK, Status.IN_PROGRESS)
            logger.info("Starting entity network mapping...")
        
            # Get or prepare L2 data
            self._prepare_l2_data()

            l2_generator = L2Generator(
                data_path=os.path.join(os.getcwd(), "resources"),
                preferred_lang=self.l2_data["basic_info"]["lang"],
            )
            l2_generator.data_preprocess(self.l2_data["notes"], self.l2_data["basic_info"])
            
            self.progress.mark_step_status(ProcessStep.MAP_ENTITY_NETWORK, Status.COMPLETED)
            logger.info("Entity network mapping completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Map entity network failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.MAP_ENTITY_NETWORK, Status.FAILED)
            self._cleanup_resources()
            return False

    def decode_preference_patterns(self)->bool:
        """Decode preference patterns using notes and related data"""
        try:
            params_manager = TrainingParamsManager()
            training_params = params_manager.get_latest_training_params()
            concurrency_threads = training_params.get("concurrency_threads")
            data_synthesis_mode = training_params.get("data_synthesis_mode")
            os.environ["CONCURRENCY_THREADS"] = str(concurrency_threads)
            os.environ["DATA_SYNTHESIS_MODE"] = data_synthesis_mode
            
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.DECODE_PREFERENCE_PATTERNS, Status.IN_PROGRESS)
            logger.info("Starting preference patterns decoding...")
            # Get or prepare L2 data
            self._prepare_l2_data()

            # Use data from l2_data dictionary
            training_params = TrainingParamsManager.get_latest_training_params()
            L2Generator(
                is_cot=training_params.get("is_cot", False),
                preferred_lang=self.l2_data["basic_info"]["lang"],
            ).gen_preference_data(
                    self.l2_data["notes"],
                    self.l2_data["basic_info"],
                    self.l2_data["data_output_base_dir"],
                    self.l2_data["topics_path"],
                    self.l2_data["entitys_path"],
                    self.l2_data["graph_path"],
                    self.l2_data["config_path"]
                    )
            
            self.progress.mark_step_status(ProcessStep.DECODE_PREFERENCE_PATTERNS, Status.COMPLETED)
            logger.info("Preference patterns decoding completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Decode preference patterns failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.DECODE_PREFERENCE_PATTERNS, Status.FAILED)
            return False

    def reinforce_identity(self)->bool:
        """Reinforce identity using notes and related data"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.REINFORCE_IDENTITY, Status.IN_PROGRESS)
            logger.info("Starting identity reinforcement...")
            # Get or prepare L2 data
            self._prepare_l2_data()

            # Get training parameters
            training_params = TrainingParamsManager.get_latest_training_params()
            # Use data from l2_data dictionary
            l2_generator = L2Generator(
                data_path=os.path.join(os.getcwd(), "resources"), is_cot=training_params.get("is_cot", False),
                preferred_lang=self.l2_data["basic_info"]["lang"],
                )  
            l2_generator.gen_selfqa_data(
                    self.l2_data["notes"],
                    self.l2_data["basic_info"],
                    self.l2_data["data_output_base_dir"],
                    self.l2_data["topics_path"],
                    self.l2_data["entitys_path"],
                    self.l2_data["graph_path"],
                    self.l2_data["config_path"]
                    )
            
            self.progress.mark_step_status(ProcessStep.REINFORCE_IDENTITY, Status.COMPLETED)
            logger.info("Identity reinforcement completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Reinforce identity failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.REINFORCE_IDENTITY, Status.FAILED)
            return False
            
    def _cleanup_resources(self):
        """Clean up resources to prevent memory leaks"""
        logger.info("Cleaning up resources to prevent memory leaks")
        
        # Clean up large data structures in l2_data dictionary
        for key in self.l2_data:
            self.l2_data[key] = None
        
        self.l2_data_prepared = False
        
        # Force garbage collection
        gc.collect()
        
        # Log memory usage after cleanup
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        logger.info(f"Memory usage after cleanup: {memory_info.rss / 1024 / 1024:.2f} MB")
    
    def augment_content_retention(self) -> bool:
        """Augment content retention using notes, basic info and graph data"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.AUGMENT_CONTENT_RETENTION, Status.IN_PROGRESS)
            logger.info("Starting content retention augmentation...")
            # Get or prepare L2 data
            self._prepare_l2_data()

            # Get training parameters
            training_params = TrainingParamsManager.get_latest_training_params()
            # Use data from l2_data dictionary
            l2_generator = L2Generator(
                data_path=os.path.join(os.getcwd(), "resources"), is_cot=training_params.get("is_cot", False),
                preferred_lang=self.l2_data["basic_info"]["lang"],
            )
            l2_generator.gen_diversity_data(
                self.l2_data["notes"],
                self.l2_data["basic_info"],
                self.l2_data["data_output_base_dir"],
                self.l2_data["topics_path"],
                self.l2_data["entitys_path"],
                self.l2_data["graph_path"],
                self.l2_data["config_path"]
            )
            l2_generator.merge_json_files(self.l2_data["data_output_base_dir"])
            # Mark step as completed
            logger.info("Content retention augmentation completed successfully")
            self.progress.mark_step_status(ProcessStep.AUGMENT_CONTENT_RETENTION, Status.COMPLETED)
            
            # Clean up resources after completion
            self._cleanup_resources()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to augment content retention: {str(e)}")
            self.progress.mark_step_status(ProcessStep.AUGMENT_CONTENT_RETENTION, Status.FAILED)
            # Clean up resources even if there was an error
            self._cleanup_resources()
            return False

    def _prepare_l2_data(self) -> dict:
        """Prepare common data needed for L2 generation tasks using lazy loading

        Returns:
            Dictionary containing all L2 data:
            - notes: List of prepared notes
            - basic_info: Dict containing user information
            - data_output_base_dir: Path to output directory
            - topics_path: Path to topics data
            - entitys_path: Path to entity mapping file
            - graph_path: Path to graph data
            - config_path: Path to config file
        """
        # If data is already prepared, return cached data directly
        if self.l2_data_prepared and all(self.l2_data.values()):
            logger.info("Using cached L2 data")
            return self.l2_data

        logger.info("Preparing L2 data...")

        # Setup directories and paths
        config = Config.from_env()
        base_dir = os.path.join(
            os.getcwd(), config.get("USER_DATA_PIPELINE_DIR") + "/raw_data"
        )
        os.makedirs(base_dir, exist_ok=True)

        # get topic
        topics_path = os.path.join(base_dir, "topics.json")
        self.l2_data["topics_path"] = topics_path
        logger.info("Topics data not found, generating it...")
        chunk_service = ChunkService()
        topics_data = chunk_service.query_topics_data()
        save_true_topics(topics_data, topics_path)

        # Initialize storage
        storage = NotesStorage()
        logger.info("Notes not found, preparing them...")
        documents = document_service.list_documents_with_l0()
        logger.info(f"list_documents_with_l0 len: {len(documents)}")
        notes_list, _ = extract_notes_from_documents(documents)
        logger.info(f"extract_notes_from_documents len: {len(notes_list)}")
        note_service = NoteService()
        note_service.prepareNotes(notes_list)
        storage.save_notes(notes_list)
        self.l2_data["notes"] = storage.load_notes()

        # Get paths
        self.l2_data["config_path"] = os.path.join(
            os.getcwd(),
            "resources/L2/data_pipeline/data_prep/subjective/config/config.json",
        )
        self.l2_data["entitys_path"] = os.path.join(
            os.getcwd(),
            "resources/L2/data_pipeline/raw_data/id_entity_mapping_subjective_v2.json",
        )
        self.l2_data["graph_path"] = os.path.join(
            os.getcwd(),
            "resources/L1/graphrag_indexing_output/subjective/entities.parquet",
        )
        self.l2_data["data_output_base_dir"] = os.path.join(os.getcwd(), "resources/L2/data")

        # Lazy load user information
        logger.info("Loading user information...")
        status_bio = get_latest_status_bio()
        global_bio = get_latest_global_bio()
        self.l2_data["basic_info"] = {
            "username": LoadService.get_current_upload_name() or "user",
            "aboutMe": LoadService.get_current_upload_description(),
            "statusBio": status_bio.content if status_bio else "",
            "globalBio": global_bio.content_third_view
                if global_bio and getattr(global_bio, "status", "") != "stale" else "",
            "lang": normalize_training_language(
                TrainingParamsManager.get_latest_training_params().get("training_language")
                or config.get("PREFER_LANGUAGE", "English")
            ),
        }

        # Mark data as prepared
        self.l2_data_prepared = True

        return self.l2_data
    def train(self) -> bool:
        """Start model training"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.TRAIN, Status.IN_PROGRESS)
            
            params = TrainingParamsManager.get_latest_training_params()
            self.training_backend = getattr(self, "training_backend", None) or resolve_training_backend(params)
            TrainingParamsManager.update_training_params({
                "resolved_training_backend": self.training_backend,
                "training_objective": "sft", "dpo_executed": False,
            })
            self._training_done.clear()
            # Get paths for the model
            paths = self._get_model_paths(self.model_name)
            
            if self.training_backend == "mlx":
                marker = os.path.join(paths["personal_dir"], "training_complete.json")
                if os.path.exists(marker):
                    os.remove(marker)
            # Check if the model directory exists and has the necessary files
            config_file = os.path.join(paths["base_path"], "config.json")
            if not os.path.exists(paths["base_path"]) or not os.path.exists(config_file):
                logger.info(f"Model '{self.model_name}' needs to be downloaded or is missing config.json")
                # Call model_download to download the model
                download_success = self.model_download()
                if not download_success:
                    logger.error(f"Failed to download model '{self.model_name}'")
                    self.progress.mark_step_status(ProcessStep.MODEL_DOWNLOAD, Status.FAILED)
                    return False
            
            # Prepare log directory and file
            log_dir = os.path.join(os.getcwd(), "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "train", "train.log")
            logger.info(f"Log file path: {log_path}")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            # A retry reads only this run, never an old 100% progress line.
            open(log_path, "w").close()
            
            # Ensure output directory exists
            os.makedirs(paths["personal_dir"], exist_ok=True)
            
            # Set USER_NAME environment variable
            os.environ["USER_NAME"] = LoadService.get_current_upload_name() or "user"
            logger.info(f"USER_NAME environment variable set: {os.environ['USER_NAME']}")
            
            script_path = os.path.join(os.getcwd(), "lpm_kernel/L2/train_for_user.sh")
            
            # First start monitoring progress in a separate thread
            logger.info("Starting monitoring thread first...")
            monitor_thread = threading.Thread(
                target=self._monitor_training_progress,
                args=(log_path,),
                daemon=True
            )
            monitor_thread.start()
            
            # Allow a moment for the monitoring thread to initialize
            time.sleep(1)
            
            # Then directly execute training process (blocking)
            logger.info("Now starting training process (blocking)...")
            training_result = self._start_training(script_path, log_path)
            
            if not training_result:
                self._training_done.set()
                monitor_thread.join(timeout=2)
                if self.is_stopped:
                    self.progress.mark_step_status(ProcessStep.TRAIN, Status.SUSPENDED)
                    return False
                logger.error("Training process failed to start")
                self.progress.mark_step_status(ProcessStep.TRAIN, Status.FAILED)
                return False
                
            # Wait for the monitoring thread to finish
            logger.info("Training process completed, waiting for monitoring to finish...")
            monitor_thread.join(timeout=10)  # Wait up to 10 seconds for monitor to finish
            
            # Check if the training was successful by checking the returncode
            if hasattr(self, 'training_result') and self.training_result:
                if self.training_result.get('returncode', 1) != 0:
                    error_msg = f"Training failed: {self.training_result.get('error', 'Unknown error')}"
                    logger.error(error_msg)
                    self.progress.mark_step_status(ProcessStep.TRAIN, Status.FAILED)
                    return False
        
            if self.is_stopped:
                self.progress.mark_step_status(ProcessStep.TRAIN, Status.SUSPENDED)
                return False
            artifact = "adapters.safetensors" if self.training_backend == "mlx" else "adapter_model.safetensors"
            if not os.path.isfile(os.path.join(paths["personal_dir"], artifact)):
                raise RuntimeError(f"Training produced no {artifact}")
            self.progress.mark_step_status(ProcessStep.TRAIN, Status.COMPLETED)
            return True

        except Exception as e:
            self._training_done.set()
            logger.error(f"Failed to start training: {str(e)}")
            self.progress.mark_step_status(ProcessStep.TRAIN, Status.FAILED)
            return False
            
    def _get_model_paths(self, model_name):
        """Get all relevant paths for a model and set environment variables
        
        Args:
            model_name: Model name
            
        Returns:
            Dictionary containing all related paths:
            - base_path: Base model path
            - personal_dir: Personal trained model output directory
            - merged_dir: Merged model output directory
            - gguf_dir: GGUF model output directory
        """
        base_dir = os.getcwd()
        paths = {
            "base_path": os.path.join(base_dir, "resources/L2/base_models", model_name),
            "personal_dir": os.path.join(base_dir, "resources/model/output/personal_model", model_name),
            "merged_dir": os.path.join(base_dir, "resources/model/output/merged_model", model_name),
            "gguf_dir": os.path.join(base_dir, "resources/model/output/gguf", model_name)
        }
        
        # Ensure all directories exist
        for path in paths.values():
            os.makedirs(path, exist_ok=True)
            
        # Set environment variables
        os.environ["MODEL_BASE_PATH"] = paths["base_path"]
        os.environ["MODEL_PERSONAL_DIR"] = paths["personal_dir"]
        os.environ["MODEL_MERGED_DIR"] = paths["merged_dir"]
        os.environ["MODEL_GGUF_DIR"] = paths["gguf_dir"]
        
        # Log environment variables
        logger.info("Set environment variables:")
        logger.info(f"MODEL_BASE_PATH: {paths['base_path']}")
        logger.info(f"MODEL_PERSONAL_DIR: {paths['personal_dir']}")
        logger.info(f"MODEL_MERGED_DIR: {paths['merged_dir']}")
        logger.info(f"MODEL_GGUF_DIR: {paths['gguf_dir']}")
        
        return paths
        
    def _start_training(self, script_path, log_path):
        """Start training process
        
        Args:
            script_path: Path to training script
            log_path: Path to log file
            
        Returns:
            bool: True if the training process started successfully, False otherwise
        """
        try:
            if self.is_stopped:
                return False
            
            # Get the latest training parameters from the class
            params_manager = TrainingParamsManager()
            training_params = params_manager.get_latest_training_params()
            learning_rate = training_params.get("learning_rate")
            num_train_epochs = training_params.get("number_of_epochs")
            concurrency_threads = training_params.get("concurrency_threads")
            data_synthesis_mode = training_params.get("data_synthesis_mode")
            use_cuda = training_params.get("use_cuda", False)
            is_cot = training_params.get("is_cot", False)
            
            # Log training parameters
            logger.info("Training parameters from latest settings:")
            logger.info(f"  Learning rate: {learning_rate}")
            logger.info(f"  Number of epochs: {num_train_epochs}")
            logger.info(f"  Concurrency threads: {concurrency_threads}")
            logger.info(f"  Data synthesis mode: {data_synthesis_mode}")
            logger.info(f"  Use CUDA: {use_cuda}")
            logger.info(f"  Is CoT: {is_cot}")
            
            # Prepare arguments for the script
            # Build command line arguments, need to include script path as the first parameter
            cmd = [
                script_path,
                "--lr", str(learning_rate),
                "--epochs", str(num_train_epochs),
                "--threads", str(concurrency_threads),
                "--mode", str(data_synthesis_mode),
                "--cuda", str(use_cuda),
                "--is_cot", str(is_cot),
                "--batch_size", str(training_params["batch_size"]),
                "--grad_accum", str(training_params["gradient_accumulation_steps"]),
                "--max_length", str(training_params["max_seq_length"]),
                "--grad_checkpoint", str(bool(training_params.get("gradient_checkpointing", True))),
            ]
            
            if getattr(self, "training_backend", None) == "mlx":
                paths = self._get_model_paths(self.model_name)
                cmd = [
                    sys.executable, "-u", "-m", "lpm_kernel.L2.mlx_training.train",
                    "--model", paths["base_path"], "--output", paths["personal_dir"],
                    "--data", os.path.join(os.getcwd(), "resources", "L2", "data", "merged.json"),
                    "--user-name", LoadService.get_current_upload_name() or "user",
                    "--lr", str(learning_rate), "--epochs", str(num_train_epochs),
                    "--batch-size", str(training_params["batch_size"]),
                    "--grad-accumulation", str(training_params["gradient_accumulation_steps"]),
                    "--max-length", str(training_params["max_seq_length"]),
                    "--validation-batches", str(training_params["validation_batches"]),
                    "--group-by-length" if training_params["group_by_length"] else "--no-group-by-length",
                ]
                if training_params.get("gradient_checkpointing"):
                    cmd.append("--grad-checkpoint")
                if training_params.get("max_steps") is not None:
                    cmd.extend(["--max-steps", str(training_params["max_steps"])])
                if is_cot:
                    cmd.append("--is-cot")

            # Ensure log directory exists
            os.makedirs(os.path.dirname(log_path), exist_ok=True)

            # Set environment variables to improve tqdm output
            env = os.environ.copy()
            env["PYTHON_EXECUTABLE"] = sys.executable
            env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
            env["PYTHONUNBUFFERED"] = "1"  # Force Python to be unbuffered
            env["FORCE_COLOR"] = "1"       # Force colored output
            env["TQDM_FORCE_TTY"] = "1"    # Force tqdm to use TTY features
            
            # Ensure log directory exists
            log_dir = os.path.dirname(log_path)
            os.makedirs(log_dir, exist_ok=True)
            
            result = self._run_command(cmd, log_path, env=env)
            return_code = result["returncode"]
            self._training_done.set()

            # Save results for train method to check
            self.training_result = {
                "returncode": return_code,
                "error": f"Execution failed, return code: {return_code}" if return_code != 0 else None
            }
            
            if return_code != 0:
                logger.error(f"Command execution failed, return code: {return_code}")
                return False
            else:
                logger.info(f"Command execution successful, return code: {return_code}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start training process: {str(e)}")
            return False

    def _monitor_training_progress(self, log_file) -> bool:
        """Read progress; process exit and saved artifacts decide completion."""
        last_position = 0
        training_started = False
        while True:
            try:
                with open(log_file, "r", encoding="utf-8", errors="replace") as stream:
                    stream.seek(last_position)
                    lines = stream.readlines()
                    last_position = stream.tell()
                for line in lines:
                    if "***** Running training *****" in line:
                        training_started = True
                    if training_started:
                        match = re.search(r"(\d+)%\|[^|]+\| (\d+)/(\d+)", line)
                        if match:
                            percentage, current, total = map(int, match.groups())
                            self._update_progress(
                                "training_to_create_second_me", "train", min(99, percentage),
                                f"Current step: {current}/{total}",
                            )
                if self._training_done.is_set() or self.is_stopped:
                    return True
            except FileNotFoundError:
                if self._training_done.is_set() or self.is_stopped:
                    return False
            time.sleep(0.1)

    def _run_command(self, command, log_path, env=None):
        """Track only this workflow's subprocess, including fuse and conversion."""
        if self.is_stopped:
            return {"returncode": -1, "error": "Training was cancelled"}
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as output:
            with self._process_lock:
                if self.is_stopped:
                    return {"returncode": -1, "error": "Training was cancelled"}
                process = subprocess.Popen(
                    command, stdout=output, stderr=subprocess.STDOUT,
                    env=env or os.environ.copy(), start_new_session=True,
                )
                self.process = process
            self.current_pid = process.pid
            return_code = process.wait()
        return {"returncode": return_code, "error": f"See {log_path}" if return_code else None}

    def _update_progress(self, stage: str, step: str, percentage: float, message: str):
        """Update progress for any stage and step"""
        try:
            self.progress.progress.update_progress(
                stage,  # stage
                step,   # step
                Status.IN_PROGRESS,
                percentage
            )
            logger.info(f"Progress updated: {percentage}% - {message}")
        except Exception as e:
            logger.error(f"Progress callback error: {str(e)}")

    def _monitor_model_download(self) -> bool:
        """Monitor model download progress"""
        try:
            # log_dir = os.path.join(os.getcwd(), "logs")
            # log_file = os.path.join(log_dir, "model_download.log")
            log_file = TRAIN_LOG_FILE
            
            # Initialize last_position to the end of file to only process new content
            try:
                with open(log_file, 'r') as f:
                    f.seek(0, 2)  # Move to the end of file
                    last_position = f.tell()
            except FileNotFoundError:
                # If file doesn't exist yet, start from beginning when it's created
                last_position = 0
            
            # Variables to track download status
            current_file = ""
            file_size = 0
            total_size = 0  # Total size of all files
            file_sizes = {}  # Dictionary to store file sizes
            last_update_time = time.time()
            
            while True:
                try:
                    # Read new log content
                    with open(log_file, 'r') as f:
                        f.seek(last_position)
                        new_lines = f.readlines()
                        last_position = f.tell()
                    
                    for line in new_lines:
                        line = line.strip()
                        
                        # Check for download start
                        if "Starting download of model:" in line:
                            logger.info("Model download started")
                            continue
                        
                        # Get file size information when a download starts
                        if "Starting download of file:" in line:
                            match = re.search(r"Starting download of file: (.+) \(Size: ([\d\.]+) MB\)", line)
                            if match:
                                current_file = match.group(1)
                                file_size = float(match.group(2))
                                file_sizes[current_file] = file_size
                                total_size = sum(file_sizes.values())
                                # logger.info(f"Starting download of {current_file} ({file_size} MB)")
                        
                        # Track file download progress
                        if "Downloaded" in line and "MB /" in line:
                            match = re.search(r"File (.+): Downloaded ([\d\.]+) MB / ([\d\.]+) MB \(([\d\.]+)%\)", line)
                            if match:
                                file_name = match.group(1)
                                downloaded_mb = float(match.group(2))
                                total_mb = float(match.group(3))
                                percentage = float(match.group(4))
                                
                                # Update file size if it was updated (especially for model.safetensors)
                                if total_mb > file_sizes.get(file_name, 0):
                                    file_sizes[file_name] = total_mb
                                    total_size = sum(file_sizes.values())
                                
                                # Calculate overall progress
                                if total_size > 0:
                                    # Sum up all downloaded data
                                    completed_files_size = sum([file_sizes.get(f, 0) for f in file_sizes if f != file_name])
                                    current_file_downloaded = (percentage / 100.0) * total_mb
                                    overall_downloaded = completed_files_size + current_file_downloaded
                                    current_progress = (overall_downloaded / total_size) * 100
                                    current_progress = min(99.0, current_progress)  # Cap at 99% until fully complete
                                    # Update progress at most once per second
                                    current_time = time.time()
                                    if current_time - last_update_time >= 3.0:

                                        self._update_progress(
                                            "downloading_the_base_model", 
                                            "model_download", 
                                            current_progress, 
                                            f"Overall: {current_progress:.1f}% - Downloading {file_name}: {percentage}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)"
                                        )
                                        last_update_time = current_time

                        if "Model downloaded successfully" in line:
                            self.progress.mark_step_status(ProcessStep.MODEL_DOWNLOAD, Status.COMPLETED)
                            logger.info("Model download completed")
                            return True
                    
                    # Briefly pause to avoid excessive CPU usage
                    time.sleep(0.1)
                    
                except IOError as e:
                    logger.error(f"Failed to read log file: {str(e)}")
                    time.sleep(0.1)
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to monitor model download progress: {str(e)}")
            return False
            
    def merge_weights(self) -> bool:
        """Merge weights"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.MERGE_WEIGHTS, Status.IN_PROGRESS)

            paths = self._get_model_paths(self.model_name)
            
            # Check if model exists
            if not os.path.exists(paths["base_path"]):
                logger.error(f"Model '{self.model_name}' does not exist, please download first")
                self.progress.mark_step_status(ProcessStep.MERGE_WEIGHTS, Status.FAILED)
                return False
            
            # Check if training output exists
            if not os.path.exists(paths["personal_dir"]):
                raise RuntimeError("Training output does not exist; train the model first")

            # Ensure merged output directory exists
            os.makedirs(paths["merged_dir"], exist_ok=True)
                
            log_path = os.path.join(os.getcwd(), "logs", f"merge_weights_{self.model_name}.log")
            
            # Ensure log directory exists
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            # Use script executor to execute merge script
            params = TrainingParamsManager.get_latest_training_params()
            backend = params.get("resolved_training_backend") or resolve_training_backend(params)
            if backend == "mlx":
                marker = os.path.join(paths["personal_dir"], "training_complete.json")
                if not os.path.isfile(marker):
                    raise RuntimeError("MLX training has not completed; refusing to fuse stale adapters")
                with open(marker, encoding="utf-8") as stream:
                    completion = json.load(stream)
                if completion.get("model") != os.path.realpath(paths["base_path"]):
                    raise RuntimeError("MLX adapter was trained against a different base model")
                command = [sys.executable, "-u", "-m", "mlx_lm", "fuse",
                           "--model", paths["base_path"], "--adapter-path", paths["personal_dir"],
                           "--save-path", paths["merged_dir"]]
            else:
                command = [
                    sys.executable, "-u", "-m", "lpm_kernel.L2.merge_lora_weights",
                    "--base_model_path", paths["base_path"],
                    "--lora_adapter_path", paths["personal_dir"],
                    "--output_model_path", paths["merged_dir"],
                    "--use-cuda" if params.get("use_cuda", False) else "--no-use-cuda",
                ]
            result = self._run_command(command, log_path)
            
            logger.info(f"Weight merge task result: {result}")
            
            # Check if script execution was successful
            if result.get('returncode', 1) != 0:
                error_msg = f"Merge weights failed: {result.get('error', 'Unknown error')}"
                logger.error(error_msg)
                self.progress.mark_step_status(ProcessStep.MERGE_WEIGHTS, Status.FAILED)
                return False
                
            # Check if merged model files exist
            config_path = os.path.join(paths["merged_dir"], "config.json")
            if not os.path.exists(config_path):
                error_msg = f"Merged model files not found in {paths['merged_dir']}"
                logger.error(error_msg)
                self.progress.mark_step_status(ProcessStep.MERGE_WEIGHTS, Status.FAILED)
                return False
            
            if self.is_stopped:
                self.progress.mark_step_status(ProcessStep.MERGE_WEIGHTS, Status.SUSPENDED)
                return False
            logger.info("Weight merge completed successfully")
            self.progress.mark_step_status(ProcessStep.MERGE_WEIGHTS, Status.COMPLETED)
            return True

        except Exception as e:
            self.progress.mark_step_status(ProcessStep.MERGE_WEIGHTS, Status.FAILED)
            logger.error(f"Merge weights failed: {str(e)}")
            return False

    def convert_model(self) -> bool:
        """Convert model to GGUF format"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.CONVERT_MODEL, Status.IN_PROGRESS)

            # Get paths for the model
            paths = self._get_model_paths(self.model_name)
            
            # Check if merged model exists
            merged_model_dir = paths["merged_dir"]
            logger.info(f"Merged model path: {merged_model_dir}")
            if not os.path.exists(merged_model_dir):
                logger.error(f"Model '{self.model_name}' merged output does not exist, please merge model first")
                self.progress.mark_step_status(ProcessStep.CONVERT_MODEL, Status.FAILED)
                return False
            
            # Get GGUF output directory
            gguf_dir = paths["gguf_dir"]
            logger.info(f"GGUF output directory: {gguf_dir}")
            
            script_path = os.path.join(os.getcwd(), "lpm_kernel/L2/convert_hf_to_gguf.py")
            gguf_path = os.path.join(gguf_dir, "model.gguf")
            logger.info(f"GGUF output path: {gguf_path}")
            
            # Build parameters
            args = [
                merged_model_dir,
                "--outfile",
                gguf_path,
                "--outtype",
                "f16",
            ]
            logger.info(f"Parameters: {args}")
            
            
            # Ensure GGUF output directory exists
            os.makedirs(os.path.dirname(gguf_path), exist_ok=True)
            
            # Use script executor to execute conversion script
            if not os.path.isfile(script_path):
                raise RuntimeError("Pinned llama.cpp converter is missing; run setup first")
            result = self._run_command(
                [sys.executable, "-u", script_path, *args],
                os.path.join(os.getcwd(), "logs", f"convert_model_{self.model_name}.log"),
            )
            
            logger.info(f"Model conversion result: {result}")
            
            # Check if script execution was successful
            if result.get('returncode', 1) != 0:
                error_msg = f"Model conversion failed: {result.get('error', 'Unknown error')}"
                logger.error(error_msg)
                self.progress.mark_step_status(ProcessStep.CONVERT_MODEL, Status.FAILED)
                return False
                
            # Check if GGUF model file exists
            if not os.path.exists(gguf_path):
                error_msg = f"GGUF model file not found at {gguf_path}"
                logger.error(error_msg)
                self.progress.mark_step_status(ProcessStep.CONVERT_MODEL, Status.FAILED)
                return False
            
            if self.is_stopped:
                self.progress.mark_step_status(ProcessStep.CONVERT_MODEL, Status.SUSPENDED)
                return False
            logger.info("Model conversion completed successfully")
            self.progress.mark_step_status(ProcessStep.CONVERT_MODEL, Status.COMPLETED)
            return True
            
        except Exception as e:
            self.progress.mark_step_status(ProcessStep.CONVERT_MODEL, Status.FAILED)
            logger.error(f"Convert model failed: {str(e)}")
            return False

    def check_training_condition(self) -> bool:
        """
        Check if the conditions for training are met
        Returns:
            bool: True if conditions are met, False otherwise
        """
        try:
            # Check if there are any documents that need embedding
            if document_service.check_all_documents_embeding_status():
                logger.warning("Cannot start training: There are documents that need embedding process first")
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking training conditions: {str(e)}", exc_info=True)
            if self.progress.progress.current_stage:
                current_step = self.progress.progress.stages[self.progress.progress.current_stage].current_step
                if current_step:
                    step = ProcessStep(current_step)
                    self.progress.mark_step_status(step, Status.FAILED)
            return False

    def start_process(self) -> bool:
        """Execute/resume the workflow and persist timings even on failure."""
        params = TrainingParamsManager.get_latest_training_params()
        metrics = PerformanceRun(metadata={
            key: params.get(key) for key in (
                "model_name", "training_backend", "concurrency_threads", "data_synthesis_mode",
                "batch_size", "gradient_accumulation_steps", "max_seq_length", "gradient_checkpointing",
                "group_by_length", "validation_batches",
            )
        })
        metrics.status = "failed"
        with metrics:
            try:
                self.is_stopped = False
                self.training_backend = resolve_training_backend(params)
                os.environ["CONCURRENCY_THREADS"] = str(get_synthesis_workers({
                    "CONCURRENCY_THREADS": str(params.get("concurrency_threads", 2))
                }))
                os.environ["DATA_SYNTHESIS_MODE"] = params.get("data_synthesis_mode") or "low"
                TrainingParamsManager.update_training_params({"resolved_training_backend": self.training_backend})
                self.current_pid = None  # Track only this workflow's subprocesses.
                self.performance_report_directory = str(metrics.directory)
                ordered_steps = ProcessStep.get_ordered_steps()
                last_successful_step = self.progress.get_last_successful_step()
                start_index = ordered_steps.index(last_successful_step) + 1 if last_successful_step else 0

                for step in ordered_steps[start_index:]:
                    self.current_step = step
                    if self.is_stopped:
                        metrics.status = "suspended"
                        self.progress.mark_step_status(step, Status.SUSPENDED)
                        return False
                    logger.info("Starting step: %s", step.value)
                    with metrics.step(step.value) as timing:
                        method = getattr(self, step.get_method_name(), None)
                        if method is None:
                            logger.error("Method %s not found", step.get_method_name())
                            self.progress.mark_step_status(step, Status.FAILED)
                            return False
                        success = method()
                        timing["status"] = (
                            "suspended" if self.is_stopped else "completed" if success else "failed"
                        )
                        if not success or self.is_stopped:
                            metrics.status = timing["status"]
                            self.progress.mark_step_status(
                                step, Status.SUSPENDED if self.is_stopped else Status.FAILED
                            )
                            return False
                    logger.info("Step %s completed successfully", step.value)
                metrics.status = "suspended" if self.is_stopped else "completed"
                return not self.is_stopped
            except Exception as exc:
                metrics.status = "suspended" if self.is_stopped else "failed"
                logger.error("Training workflow failed: %s", exc, exc_info=True)
                if getattr(self, "current_step", None):
                    self.progress.mark_step_status(
                        self.current_step, Status.SUSPENDED if self.is_stopped else Status.FAILED
                    )
                return False

    def reset_progress(self):
        """Save current progress
        
        This method saves the current progress to the progress file.
        """
        try:
            self.progress.reset_progress()
            logger.info("Progress saved successfully")
        except Exception as e:
            logger.error(f"Failed to save progress: {str(e)}", exc_info=True)
            
    def get_step_output_content(self, step_name: str = None) -> Optional[Dict]:
        """Get content of output file for a specific training step
        
        Args:
            step_name: Name of the step to get content for. Required parameter.
        
        Returns:
            Optional[Dict]: Content of the output file for the specified step, or None if not found
        """
        try:
            if step_name == "generate_biography":
                logger.info("Querying L1 version data for biography")
                return query_l1_version_data(1)

            # If step_name is not provided or invalid, return None
            if not step_name or step_name not in output_files:
                return None
            
            # Get file path for the requested step
            file_path = output_files[step_name]
            if not os.path.exists(file_path):
                return None
            
            # Read and return file content
            return read_file_content(file_path)
        except Exception as e:
            logger.error(f"Error getting step output content: {str(e)}")
            return None

    def stop_process(self):
        """Cancel this workflow's subprocess group, never the backend's other children."""
        self.is_stopped = True
        self._training_done.set()
        if self.current_step:
            self.progress.mark_step_status(self.current_step, Status.SUSPENDED)
        with self._process_lock:
            process = getattr(self, "process", None)
        if process is None or process.poll() is not None:
            return True
        try:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=3)
            return True
        except ProcessLookupError:
            return True
        except OSError:
            logger.exception("Failed to stop training subprocess group")
            return False
