from typing import Dict, List, Optional, Union, Any
import logging

from lpm_kernel.common.performance import create_synthesis_client

from lpm_kernel.L1.bio import Bio, Chat, Note, Todo, UserInfo
from lpm_kernel.L1.prompt import PREFER_LANGUAGE_SYSTEM_PROMPT, STATUS_BIO_SYSTEM_PROMPT
from lpm_kernel.L1.utils import get_cur_time, is_valid_chat, is_valid_note, is_valid_todo
from lpm_kernel.api.services.user_llm_config_service import UserLLMConfigService
from lpm_kernel.configs.config import Config
from lpm_kernel.configs.logging import get_train_process_logger

logger = get_train_process_logger()


class StatusBioGenerator:
    def __init__(self):
        self.preferred_language = "English"
        self.model_params = {
            "temperature": 0,
            "max_tokens": 1000,
            "top_p": 0,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "seed": 42,
        }
        self.user_llm_config_service = UserLLMConfigService()
        self.user_llm_config = self.user_llm_config_service.get_available_llm()
        if self.user_llm_config is None:
            self.client = None
            self.model_name = None
        else:
            self.client = create_synthesis_client(
                "l1_status",
                api_key=self.user_llm_config.chat_api_key,
                base_url=self.user_llm_config.chat_endpoint,
                timeout=180.0,  # Set global timeout
            )
            self.model_name = self.user_llm_config.chat_model_name
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
            self.model_params["top_p"] = 0.001
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
                **self.model_params,
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
                        **self.model_params,
                        **kwargs
                    )
            
            # Re-raise the exception
            raise

    def _build_message(self, user_info: UserInfo, language: str) -> List[Dict[str, str]]:
        """Build message list for generating status biography.

        Args:
            user_info: User information object.
            language: Preferred language.

        Returns:
            List of messages formatted for LLM API.
        """
        messages = [
            {"role": "system", "content": STATUS_BIO_SYSTEM_PROMPT},
            {"role": "user", "content": str(user_info)},
        ]

        if language:
            messages.append(
                {
                    "role": "system",
                    "content": PREFER_LANGUAGE_SYSTEM_PROMPT.format(language=language),
                }
            )

        return messages


    def generate_status_bio(self, notes: List[Note], todos: List[Todo], 
                           chats: List[Chat]) -> Bio:
        """Generate a status biography based on user's notes, todos, and chats.

        Args:
            notes: List of user's notes.
            todos: List of user's todos.
            chats: List of user's chats.

        Returns:
            Bio object containing generated content.
        """
        cur_time = get_cur_time()

        user_info = UserInfo(cur_time, notes, todos, chats)
        messages = self._build_message(user_info, self.preferred_language)

        answer = self._call_llm_with_retry(messages)
        content = answer.choices[0].message.content
        logger.info(f"Generated content: {content}")

        # Create and return Bio object, ensuring all content fields have values
        return Bio(
            contentThirdView=content,  # Put generated content in third_view
            content=content,  # Put generated content in second_view
            summaryThirdView=content,  # Put generated content in third_view
            summary=content,  # Put generated content in second_view
            attributeList=[],
            shadesList=[],
        )
