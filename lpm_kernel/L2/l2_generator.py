"""L2 Generator module for handling L2 level data processing and model operations.

This module provides the L2Generator class which is responsible for data preprocessing,
subjective data generation, model conversion, and inference with the trained model.
"""

from typing import Dict, List
import os


from lpm_kernel.L1.bio import Note
from lpm_kernel.L2.data import L2DataProcessor
import yaml
import logging
from lpm_kernel.L2.data_pipeline.data_prep.preference.preference_QA_generate import PreferenceQAGenerator
from lpm_kernel.L2.data_pipeline.data_prep.diversity.diversity_data_generator import DiversityDataGenerator
from lpm_kernel.L2.data_pipeline.data_prep.selfqa.selfqa_generator import SelfQA
import json

class L2Generator:
    """L2 level generator for handling data and model operations.
    
    This class manages operations related to L2 processing, including data preprocessing,
    subjective data generation, model conversion, and model inference.
    
    Attributes:
        data_path: Path to the raw data directory.
        data_processor: Instance of L2DataProcessor for handling data processing.
    """

    def __init__(self, data_path: str = "../raw_data", preferred_lang: str = "English", is_cot: bool = True):
        """Initialize the L2Generator with data path and preferred language.
        
        Args:
            data_path: Path to the raw data directory. Defaults to "../raw_data".
            preferred_lang: Preferred language for data processing. Defaults to "English".
            is_cot: Whether to use Chain of Thought reasoning. Can be bool or string.
        """
        self.data_path = data_path
        self.data_processor = L2DataProcessor(data_path, preferred_lang)
        self.preferred_lang = preferred_lang
        
        # Convert is_cot to bool if it's a string
        if isinstance(is_cot, str):
            self.is_cot = is_cot.lower() == 'true'
        else:
            self.is_cot = bool(is_cot)
            
        logging.info(f"L2Generator initialized with is_cot={self.is_cot}")
        
    def data_preprocess(self, note_list: List[Note], basic_info: Dict):
        """Preprocess the input notes and basic information.
        
        Args:
            note_list: List of Note objects to process.
            basic_info: Dictionary containing basic user information.
        """
        self.data_processor(note_list, basic_info)

    def gen_subjective_data(
        self,
        note_list: List[Note],
        basic_info: Dict,
        data_output_base_dir: str,
        topics_path: str,
        entities_path: str,
        graph_path: str,
        config_path: str,
    ):
        """Generate subjective data for personalization.
        
        This method orchestrates the generation of subjective data including preferences,
        diversity, self-Q&A data, and graph indexing.
        
        Args:
            note_list: List of Note objects to process.
            basic_info: Dictionary containing user information.
            data_output_base_dir: Base directory for output data.
            topics_path: Path to topics data.
            entities_path: Path to entity data.
            graph_path: Path to graph data.
            config_path: Path to configuration file.
        """
        if not os.path.exists(data_output_base_dir):
            os.makedirs(data_output_base_dir)

        # Check if the file exists
        if not os.path.exists(topics_path):
            # Create an empty file
            with open(topics_path, "w") as f:
                f.write(json.dumps([]))

        # Generate subjective data
        self.data_processor.gen_subjective_data(
            note_list=note_list,
            data_output_base_dir=data_output_base_dir,
            preference_output_path="preference.json",
            diversity_output_path="diversity.json",
            selfqa_output_path="selfqa.json",
            global_bio=basic_info["globalBio"],
            topics_path=topics_path,
            entitys_path=entities_path,
            graph_path=graph_path,
            user_name=basic_info["username"],
            config_path=config_path,
            user_intro=basic_info["aboutMe"],
        )
        
        # Merge JSON files for training
        self.merge_json_files(data_output_base_dir)
        
        # Release Ollama models from memory after data synthesis is complete
        self._release_ollama_models()

    def gen_preference_data(
        self,
        note_list: List[Note],
        basic_info: Dict,
        data_output_base_dir: str,
        topics_path: str,
        entities_path: str,
        graph_path: str,
        config_path: str,
    ):
        global_bio = basic_info["globalBio"]
        preference_output_path = os.path.join(data_output_base_dir, "preference.json")

        processor = PreferenceQAGenerator(
            filename=topics_path, bio=global_bio, preference_language=self.preferred_lang, is_cot=self.is_cot
        )
        processor.process_clusters(preference_output_path)
    
    def gen_diversity_data(
        self, 
        note_list: List[Note], 
        basic_info: Dict, 
        data_output_base_dir: str, 
        topics_path: str, 
        entities_path: str, 
        graph_path: str, 
        config_path: str
    ):
        global_bio = basic_info["globalBio"]
        user_name = basic_info["username"]
        output_path = os.path.join(data_output_base_dir, "diversity.json")

        processor = DiversityDataGenerator(self.preferred_lang, is_cot=self.is_cot)
        processor.generate_data(
            entities_path, note_list, config_path, graph_path, user_name, global_bio, output_path
        )

    def gen_selfqa_data(
        self, 
        note_list: List[Note], 
        basic_info: Dict, 
        data_output_base_dir: str, 
        topics_path: str, 
        entities_path: str, 
        graph_path: str, 
        config_path: str
    ):
        global_bio = basic_info["globalBio"]
        user_name = basic_info["username"]
        user_intro = basic_info["aboutMe"]
        output_path = os.path.join(data_output_base_dir, "selfqa.json")

        selfqa = SelfQA(
            user_name=user_name,
            user_input_introduction=user_intro,
            user_global_bio= global_bio,
            preferred_language=self.preferred_lang,
            is_cot=self.is_cot
        )
        q_a_list = selfqa.generate_qa()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(q_a_list, f, ensure_ascii=False, indent=4)
    
    def merge_json_files(self, data_output_base_dir: str):
        preference_output_path = os.path.join(data_output_base_dir, "preference.json")
        diversity_output_path = os.path.join(data_output_base_dir, "diversity.json")
        selfqa_output_path = os.path.join(data_output_base_dir, "selfqa.json")

        json_files_to_merge = [
                preference_output_path,
                diversity_output_path,
                selfqa_output_path,
        ]

        merged_data = []

        for file_path in json_files_to_merge:
            if file_path and os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_data = json.load(f)
                        if isinstance(file_data, list):
                            for sample in file_data:
                                if isinstance(sample, dict):
                                    sample = dict(sample)
                                    if sample.get("memory_format_version") != 1 and sample.get("training_type") not in {"style", "values", "preference"}:
                                        logging.warning("Skipping legacy sample from %s; regenerate to attach bounded references", os.path.basename(file_path))
                                        continue
                                    if sample.get("training_type") not in {"style", "values", "preference", "retrieval_no_hit"} and not (sample.get("context") or sample.get("references")):
                                        logging.warning("Skipping factual sample without references in %s", os.path.basename(file_path))
                                        continue
                                    sample.setdefault("training_type", os.path.basename(file_path).replace(".json", ""))
                                    merged_data.append(sample)
                        else:
                            logging.warning("Skipping non-list legacy training file %s", os.path.basename(file_path))
                except Exception as e:
                    logging.error(f"Error merging file {file_path}: {str(e)}")

        # Explicit no-hit examples retain the same reference format as inference.
        chinese = self.preferred_lang.lower() in {"chinese", "中文", "zh", "cn"}
        missing = [
            ("我下一次预约的具体时间是什么？", "目前没有相关记录，无法确认具体时间。"),
            ("我去年旅行住的酒店叫什么？", "我没有找到相关记录，不能确定酒店名称。"),
        ] if chinese else [
            ("What is the exact time of my next appointment?", "I have no relevant record, so I cannot confirm the time."),
            ("What hotel did I stay at on last year's trip?", "I could not find a relevant record, so I do not know the hotel name."),
        ]
        merged_data.extend({"user": question, "assistant": answer,
                            "context": "", "training_type": "retrieval_no_hit", "memory_format_version": 1}
                           for question, answer in missing)
        from collections import Counter
        logging.info("L2 training sample types: %s", dict(Counter(
            sample.get("training_type", "unknown") for sample in merged_data)))

        # Save the merged data
        merged_output_path = os.path.join(data_output_base_dir, "merged.json")
        with open(merged_output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    def _release_ollama_models(self):
        """Release Ollama models from memory to free up VRAM for training.
        
        This method calls the release function defined in the train module.
        It's important to release models after data synthesis and before training
        to ensure VRAM is properly freed.
        """
        try:
            from lpm_kernel.L2.train import release_ollama_models
            release_ollama_models()
        except Exception as e:
            import logging
            logging = logging.getLogger(__name__)
            logging.warning(f"Failed to release Ollama models: {str(e)}")

    def clean_graphrag_keys(self):
        """Compatibility hook: credentials are no longer written to GraphRAG files."""
        logging.info("GraphRAG uses child-process credentials; no settings cleanup needed")
