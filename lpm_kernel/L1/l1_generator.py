from copy import deepcopy
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional
import logging
import os

from lpm_kernel.common.performance import create_synthesis_client

from lpm_kernel.L1.bio import (
    Bio,
    CONFIDENCE_LEVELS_INT,
    Chat,
    Cluster,
    Memory,
    Note,
    ShadeInfo,
    ShadeMergeInfo,
    Todo,
)
from lpm_kernel.L1.prompt import (
    COMMON_PERSPECTIVE_SHIFT_SYSTEM_PROMPT,
    GLOBAL_BIO_SYSTEM_PROMPT,
    PREFER_LANGUAGE_SYSTEM_PROMPT,
    SHADE_MERGE_DEFAULT_SYSTEM_PROMPT,
)
from lpm_kernel.L1.shade_generator import ShadeGenerator, ShadeMerger
from lpm_kernel.L1.status_bio_generator import StatusBioGenerator
from lpm_kernel.L1.topics_generator import TopicsGenerator
from lpm_kernel.api.services.user_llm_config_service import UserLLMConfigService
from lpm_kernel.configs.config import Config
from lpm_kernel.configs.logging import get_train_process_logger

logger = get_train_process_logger()

DATE_TIME_FORMAT = "%Y-%m-%d"


class ConfidenceLevel(str, Enum):
    VERY_LOW = "VERY LOW"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY HIGH"


IMPORTANCE_TO_CONFIDENCE = {
    1: ConfidenceLevel.VERY_LOW,
    2: ConfidenceLevel.LOW,
    3: ConfidenceLevel.MEDIUM,
    4: ConfidenceLevel.HIGH,
    5: ConfidenceLevel.VERY_HIGH,
}


class DailyTimeline:
    def __init__(self, id: int, dateTime: str, content: str, noteIds: List[int]):
        self.id = id
        self.date_time = dateTime
        self.content = content.strip()
        self.note_ids = noteIds


    def _desc_(self) -> str:
        """Returns a string representation of the daily timeline.
        
        Returns:
            str: Formatted string representation.
        """
        return f"- [{self.date_time}] {self.content}".strip()


    def to_dict(self) -> Dict[str, Any]:
        """Converts the DailyTimeline object to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the DailyTimeline.
        """
        return {
            "id": self.id,
            "dateTime": self.date_time,
            "content": self.content,
            "noteIds": self.note_ids,
        }


class MonthlyTimeline:
    def __init__(
        self, id: int, monthDate: str, title: str, dailyTimelines: List[Dict[str, Any]]
    ):
        self.id = id
        self.month_date = monthDate
        self.title = title
        daily_timelines = [
            DailyTimeline(**daily_timeline) for daily_timeline in dailyTimelines
        ]
        self.daily_timelines = sorted(
            daily_timelines,
            key=lambda x: datetime.strptime(x.date_time, DATE_TIME_FORMAT),
        )


    def _desc_(self) -> str:
        """Returns a string representation of the monthly timeline.
        
        Returns:
            str: Formatted string representation.
        """
        return f"** {self.month_date} **\n" + "\n".join(
            [daily_timeline._desc_() for daily_timeline in self.daily_timelines]
        )


    def _preview_(self, preview_num: int = 0) -> str:
        """Generates a preview of the monthly timeline.
        
        Args:
            preview_num: Number of daily timelines to include in the preview.
            
        Returns:
            str: Preview string of the monthly timeline.
        """
        preview_statement = f"[{self.month_date}] {self.title}\n"
        for daily_timeline in self.daily_timelines[:preview_num]:
            preview_statement += daily_timeline._desc_() + "\n"
        return preview_statement


    def to_dict(self) -> Dict[str, Any]:
        """Converts the MonthlyTimeline object to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the MonthlyTimeline.
        """
        return {
            "id": self.id,
            "monthDate": self.month_date,
            "title": self.title,
            "dailyTimelines": [
                daily_timeline.to_dict() for daily_timeline in self.daily_timelines
            ],
        }


class EntityWiki:
    def __init__(self, wikiText: str, monthlyTimelines: List[Dict[str, Any]]):
        self.wiki_text = wikiText
        self.monthly_timelines = [
            MonthlyTimeline(**monthly_timeline) for monthly_timeline in monthlyTimelines
        ]
        self.max_month_idx = (
            max([monthly_timeline.id for monthly_timeline in self.monthly_timelines])
            if self.monthly_timelines
            else 0
        )


    def to_dict(self) -> Dict[str, Any]:
        """Converts the EntityWiki object to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the EntityWiki.
        """
        return {
            "wikiText": self.wiki_text,
            "monthlyTimelines": [
                monthly_timeline.to_dict()
                for monthly_timeline in self.monthly_timelines
            ],
        }


class L1Generator:
    def __init__(self):
        self.preferred_language = "English"
        self.bio_model_params = {
            "temperature": 0,
            "max_tokens": 2000,
            "top_p": 0,
            "frequency_penalty": 0,
            "seed": 42,
            "presence_penalty": 0,
            "timeout": 180,
        }
        self.user_llm_config_service = UserLLMConfigService()
        self.user_llm_config = self.user_llm_config_service.get_available_llm()
        if self.user_llm_config is None:
            self.client = None
            self.model_name = None
        else:
            self.client = create_synthesis_client(
                "l1_biography",
                api_key=self.user_llm_config.chat_api_key,
                base_url=self.user_llm_config.chat_endpoint,
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
            self.bio_model_params["top_p"] = 0.001
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
                **self.bio_model_params,
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
                        **self.bio_model_params,
                        **kwargs
                    )
            
            # Re-raise the exception
            raise

    def __build_message(
        self, system_prompt: str, user_prompt: str, language: str
    ) -> List[Dict[str, str]]:
        """Builds message for LLM API call.
        
        Args:
            system_prompt: System prompt content.
            user_prompt: User prompt content.
            language: Preferred language for the response.
            
        Returns:
            List[Dict[str, str]]: Formatted message for the LLM.
        """
        raw_message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if language:
            raw_message.append(
                {
                    "role": "system",
                    "content": PREFER_LANGUAGE_SYSTEM_PROMPT.format(language=language),
                }
            )
        return raw_message


    def _global_bio_generate(self, global_bio: Bio) -> Bio:
        """Generates global biography.
        
        Args:
            global_bio: Bio object to generate content for.
            
        Returns:
            Bio: Updated Bio object with generated content.
        """
        user_prompt = global_bio.to_str()

        system_prompt = GLOBAL_BIO_SYSTEM_PROMPT

        global_bio_message = self.__build_message(
            system_prompt, user_prompt, language=self.preferred_language
        )

        response = self._call_llm_with_retry(global_bio_message)
        third_perspective_result = response.choices[0].message.content
        global_bio.summary_third_view = third_perspective_result
        global_bio.content_third_view = global_bio.complete_content()
        global_bio = self._shift_perspective(global_bio)
        global_bio = self._assign_confidence_level(global_bio)

        return global_bio


    def _shift_perspective(self, global_bio: Bio) -> Bio:
        """Shifts the perspective of the biography to second person.
        
        Args:
            global_bio: Bio object to shift perspective for.
            
        Returns:
            Bio: Updated Bio object with shifted perspective.
        """
        system_prompt = COMMON_PERSPECTIVE_SHIFT_SYSTEM_PROMPT
        user_prompt = global_bio.summary_third_view

        shift_perspective_message = self.__build_message(
            system_prompt, user_prompt, language=self.preferred_language
        )

        response = self._call_llm_with_retry(shift_perspective_message)
        second_perspective_result = response.choices[0].message.content

        global_bio.summary_second_view = second_perspective_result
        global_bio.content_second_view = global_bio.complete_content(second_view=True)
        return global_bio


    def _assign_confidence_level(self, global_bio: Bio) -> Bio:
        """Assigns confidence levels to shades in the biography.
        
        Args:
            global_bio: Bio object to assign confidence levels to.
            
        Returns:
            Bio: Updated Bio object with confidence levels assigned.
        """
        level_n, interest_n = len(IMPORTANCE_TO_CONFIDENCE), len(global_bio.shades_list)
        level_list = [
            IMPORTANCE_TO_CONFIDENCE[level_n - int(i / interest_n * level_n)]
            for i in range(interest_n)
        ]
        for shade, level in zip(global_bio.shades_list, level_list):
            shade.confidence_level = level
        return global_bio


    def gen_global_biography(
        self, old_profile: Bio, cluster_list: List[Cluster]
    ) -> Bio:
        """Generates the global biography of the user.
        
        Args:
            old_profile: Previous Bio object.
            cluster_list: List of clusters for reference.
            
        Returns:
            Bio: Updated global biography.
        """
        global_bio = deepcopy(old_profile)
        global_bio = self._global_bio_generate(global_bio)
        return global_bio


    def gen_shade_for_cluster(
        self,
        old_memory_list: List[Note],
        new_memory_list: List[Note],
        shade_info_list: List[ShadeInfo],
    )-> Optional[ShadeInfo]:
        """Generates shade for a cluster.
        
        Args:
            old_memory_list: List of previous notes.
            new_memory_list: List of new notes.
            shade_info_list: List of shade information.
            
        Returns:
            Generated shade.
        """
        shade_generator = ShadeGenerator()

        shade = shade_generator.generate_shade(
            old_memory_list=old_memory_list,
            new_memory_list=new_memory_list,
            shade_info_list=shade_info_list,
        )
        return shade


    def merge_shades(self, shade_info_list: List[ShadeMergeInfo]):
        """Merges multiple shades.
        
        Args:
            shade_info_list: List of shade merge information.
            
        Returns:
            Merged shade result.
        """
        shade_merger = ShadeMerger()
        return shade_merger.merge_shades(shade_info_list)


    def gen_status_biography(
        self, cur_time: str, notes: List[Note], todos: List[Todo], chats: List[Chat]
    ):
        """Generates the status biography of the user.
        
        Args:
            cur_time: Current time string.
            notes: List of notes.
            todos: List of todos.
            chats: List of chats.
            
        Returns:
            Generated status biography.
        """
        status_bio_generator = StatusBioGenerator()
        return status_bio_generator.generate_status_bio(notes, todos, chats)


    def gen_topics_for_shades(
        self,
        old_cluster_list: List[Cluster],
        old_outlier_memory_list: List[Memory],
        new_memory_list: List[Memory],
        cophenetic_distance: float = 1.0,
        outlier_cutoff_distance: float = 0.5,
        cluster_merge_distance: float = 0.5,
    ):
        """Generates topics for shades.
        
        Args:
            old_cluster_list: List of previous clusters.
            old_outlier_memory_list: List of previous outlier memories.
            new_memory_list: List of new memories.
            cophenetic_distance: Distance threshold for cophenetic clustering.
            outlier_cutoff_distance: Distance threshold for outlier detection.
            cluster_merge_distance: Distance threshold for cluster merging.
            
        Returns:
            Generated topics for shades.
        """
        topics_generator = TopicsGenerator()
        return topics_generator.generate_topics_for_shades(
            old_cluster_list,
            old_outlier_memory_list,
            new_memory_list,
            cophenetic_distance,
            outlier_cutoff_distance,
            cluster_merge_distance,
        )


    def generate_topics(self, notes_list: List[Note]):
        """Generates topics from a list of notes.
        
        Args:
            notes_list: List of notes to generate topics from.
            
        Returns:
            Generated topics.
        """
        topics_generator = TopicsGenerator()
        return topics_generator.generate_topics(notes_list)
