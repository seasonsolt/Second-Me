"""
Training parameters management module.
This module provides functions for managing and accessing training parameters.
"""

import logging
import json
import os

# Configure logger
logger = logging.getLogger(__name__)


class TrainingParamsManager:
    """
    Training parameters manager class.
    """
    
    # Default training parameters
    _default_training_params = {
        "model_name": "Qwen3-1.7B",
        "learning_rate": 1e-4,
        "number_of_epochs": 3,
        "concurrency_threads": 2,
        "data_synthesis_mode": "low",
        "use_cuda": False,  # Default to using CUDA when available
        "is_cot": False,
        "training_backend": "auto",
        "resolved_training_backend": None,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_seq_length": 4096,
        "gradient_checkpointing": True,
        "group_by_length": True,
        "validation_batches": 4,
        "max_steps": None,
        "training_language": None,
        "training_objective": "sft",
        "dpo_executed": False
    }
    
    # Parameters file path
    _params_file_path = None
    
    @classmethod
    def _get_params_file_path(cls):
        """
        Get the training parameters file path
        """
        if cls._params_file_path is None:
            # Set the parameters file path
            progress_dir = os.path.join(os.getcwd(), "data", "progress")
            if not os.path.exists(progress_dir):
                os.makedirs(progress_dir)
            cls._params_file_path = os.path.join(progress_dir, "training_params.json")
        
        return cls._params_file_path
    
    @classmethod
    def prepare_training_params(cls, params, use_previous_params=True):
        """Resolve the exact saved settings before API backend preflight."""
        current_params = cls.get_latest_training_params() if use_previous_params else cls._default_training_params.copy()
        for key, value in params.items():
            if key in cls._default_training_params:
                current_params[key] = cls._default_training_params[key] if value is None else value
            else:
                logger.warning(f"Ignoring unknown parameter: {key}")
        if current_params.get("is_cot"):
            # Training uses the native non-thinking template and strips reasoning.
            logger.warning("Ignoring is_cot=True: thinking-model training is no longer supported")
            current_params["is_cot"] = False
        cls.validate_training_params(current_params)
        return current_params

    # Inclusive ranges shared with the training UI.
    _integer_param_ranges = {
        "batch_size": (1, 16),
        "gradient_accumulation_steps": (1, 64),
        "max_seq_length": (128, 32768),
        "max_steps": (1, None),
    }

    @classmethod
    def validate_training_params(cls, params):
        """Raise ValueError for numeric settings the training scripts cannot use."""
        for key, (minimum, maximum) in cls._integer_param_ranges.items():
            value = params.get(key)
            if key == "max_steps" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{key} must be an integer, got {value!r}")
            if value < minimum or (maximum is not None and value > maximum):
                upper = maximum if maximum is not None else "unbounded"
                raise ValueError(f"{key} must be between {minimum} and {upper}, got {value}")

    @classmethod
    def update_training_params(cls, params, use_previous_params=True):
        """
        Update the latest training parameters and save to file
        
        Args:
            params: Dictionary containing training parameters
            use_previous_params: Whether to use previous training parameters as base
        """
        current_params = cls.prepare_training_params(params, use_previous_params)

        # Save to file
        params_file = cls._get_params_file_path()
        try:
            with open(params_file, 'w', encoding='utf-8') as f:
                json.dump(current_params, f, indent=2)
            logger.info(f"Training parameters saved to {params_file}")
        except Exception as e:
            logger.error(f"Failed to save training parameters to file: {str(e)}", exc_info=True)
    
    @classmethod
    def get_latest_training_params(cls):
        """
        Get the latest training parameters from file
        
        Returns:
            dict: Dictionary containing the latest training parameters
        """
        params_file = cls._get_params_file_path()
        
        # If file exists, read from file
        if os.path.exists(params_file):
            try:
                with open(params_file, 'r', encoding='utf-8') as f:
                    params = json.load(f)
                
                # Replace null values with default values
                default_params = cls._default_training_params.copy()
                for key, value in default_params.items():
                    if key not in params or params[key] is None:
                        params[key] = value
                
                logger.debug(f"Loaded training parameters from {params_file}")
                return params
            except Exception as e:
                logger.error(f"Failed to load training parameters from file: {str(e)}", exc_info=True)
                # If reading fails, return default parameters
                return cls._default_training_params.copy()
        else:
            # If file does not exist, return default parameters
            return cls._default_training_params.copy()