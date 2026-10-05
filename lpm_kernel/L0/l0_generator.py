
from typing import Any, Dict, List
import copy
import json
import os
import time
import traceback

from openai import OpenAI
import tiktoken

from lpm_kernel.api.services.user_llm_config_service import UserLLMConfigService
from lpm_kernel.configs.config import Config
from lpm_kernel.L0.models import InsighterInput, SummarizerInput
from lpm_kernel.L0.prompt import *
from lpm_kernel.utils import (
    DataType,
    IntentType,
    TokenParagraphSplitter,
    TokenTextSplitter,
    cal_upperbound,
    chunk_filter,
    equidistant_filter, 
    get_safe_content_turncate,
    get_summarize_title_keywords,
    select_language_desc,
)

from lpm_kernel.configs.logging import get_train_process_logger
logger = get_train_process_logger()

class L0Generator:
    def __init__(self, preferred_language="English"):
        """Initialize L0Generator with language preference.
        
        Args:
            preferred_language: The language to use for generation, defaults to English.
        """
        self.preferred_language = preferred_language

        # Initialize tokenizer
        self._tokenizer = tiktoken.get_encoding("cl100k_base")  # OpenAI default tokenizer

        self.lf_prompt_image_parser = insight_image_parser
        self.lf_prompt_image_overview = insight_image_overview
        self.lf_prompt_image_breakdown = insight_image_breakdown

        self.lf_prompt_audio_parser = insight_audio_parser
        self.lf_prompt_audio_overview = insight_audio_overview
        self.lf_prompt_audio_breakdown = insight_audio_breakdown

        self.lf_prompt_doc_overview = insight_doc_overview
        self.lf_prompt_doc_breakdown = insight_doc_breakdown

        self.max_retries_summarize = 2
        self.timeout_summarize = 180

        self.user_llm_config_service = UserLLMConfigService()
        self.user_llm_config = self.user_llm_config_service.get_available_llm()
        if self.user_llm_config is None:
            self.client = None
            self.model_name = None
        else:
            self.client = OpenAI(
                api_key=self.user_llm_config.chat_api_key,
                base_url=self.user_llm_config.chat_endpoint,
            )
            self.model_name = self.user_llm_config.chat_model_name
        

    def _insighter_image(
        self, bio: Dict[str, str], content: str, max_retries: int, request_timeout: int, file_content: str
    ) -> tuple[str, str]:
        """Process image content to generate insights.
        
        Args:
            bio: Dictionary containing user biography information
            content: Text content related to the image
            max_retries: Maximum number of API call retries
            request_timeout: Timeout for API calls in seconds
            file_content: URL or base64 content of the image
            
        Returns:
            Tuple of (summary, title) strings
        """
        hint_prompt = f"# Hint #\n{content}\n# Instruction #\n"
        language_desc = select_language_desc(self.preferred_language)

        segment_list = [
            self.lf_prompt_image_parser,
            self.lf_prompt_image_overview,
            self.lf_prompt_image_breakdown,
        ]
        messages_list = []

        for i in range(len(segment_list)):
            image_parser_prompt = segment_list[i]
            if "__global_bio__" in image_parser_prompt:
                image_parser_prompt = image_parser_prompt.replace(
                    "__about_me__", bio["about_me"]
                )
                image_parser_prompt = image_parser_prompt.replace(
                    "__global_bio__", bio["global_bio"]
                )
                image_parser_prompt = image_parser_prompt.replace(
                    "__status_bio__", bio["status_bio"]
                )

            # system prompt
            language = language_desc if i != 0 else "English"

            messages = [
                {"role": "system", "content": image_parser_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": hint_prompt
                            + "Here are some images and their Hint. Please follow the WorkFlow and do your best. Ensure that your response is in a parseable JSON format."
                            + language,
                        }
                    ],
                },
            ]

            if i == 0:
                new_messages = copy.deepcopy(messages)
                new_messages[-1]["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": file_content,  # file_content is the image url
                        },
                    }
                )
                messages_list.append(new_messages)
            else:
                messages[-1]["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": file_content,  # file_content is the image url
                        },
                    }
                )
                messages_list.append(messages)

        results = []

        for messages in messages_list:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=4096,
                temperature=0.0,
                max_retries=max_retries,
                timeout=request_timeout,
                response_format={"type": "json_object"},
            )
            results.append(response.choices[0].message.content)

        try:
            images_intent_list = []
            for image_id in range(len(results) - 2):
                images_intent_list.append(results[image_id]["image"].get("Step 3", ""))

            title = results[-2].get("Title", "")
            opening = results[-2].get("Opening", "")
            insight = results[-1].get("Insight", [])

            insight = "- " + "\n- ".join(insight) if insight else ""
            summary = "\n\n".join([opening, insight])

            return summary, title
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise RuntimeError(f"Unexpected error: {e}")

    def _insighter_audio(
        self, bio: str, content: str, max_retries: int, request_timeout: int, file_content: Dict[str, Any]
    ) -> tuple[str, str]:
        """Process audio content to generate insights.
        
        Args:
            bio: User biography information
            content: Text content related to the audio
            max_retries: Maximum number of API call retries
            request_timeout: Timeout for API calls in seconds
            file_content: Dictionary containing audio metadata and content
            
        Returns:
            Tuple of (insight, title) strings
        """
        user_info = """# Hint #
                    "{content}"

                    # Speech #
                    "{speech}"

                    # User Instruction #
                    '{user_input}'
                    """

        user_input = "Here are some speech and their hint. Please follow the WorkFlow and do your best. Ensure that your response is in a parseable JSON format. "

        language_desc = select_language_desc(self.preferred_language)
        speech_dict = file_content["metadata"]["audio"].get("segmentList", [])

        speech = ""
        end_point = 0

        # Raise exception if speech is empty or too short
        if not speech_dict:
            raise ValueError("Invalid input: speech must not be empty")

        for segment in speech_dict:
            start_time = int(segment["segmentStartTime"])
            end_time = int(segment["segmentEndTime"])
            segment_content = segment["segmentContent"]
            tmp = f"[{start_time}-{end_time}]: {segment_content}\n"
            speech += tmp
            end_point = int(end_time)
        logger.info(f"length of speech: {end_point}")

        # Split speech over 1200s into segments, maximum 1200s each
        num_segments = 1
        if end_point > 1200:
            num_segments = max(2, int(round(end_point / 1200.0)))
            segment_duration = end_point / num_segments
            speech_segments = ["" for _ in range(num_segments)]
            for segment in speech_dict:
                start_time = int(segment["segmentStartTime"])
                end_time = int(segment["segmentEndTime"])
                segment_content = segment["segmentContent"]

                segment_index = min(
                    num_segments - 1, int(start_time // segment_duration)
                )
                speech_segments[
                    segment_index
                ] += f"[{start_time}-{end_time}]: {segment_content}\n"

            user_info_overall = user_info.format(
                content=content, speech=speech, user_input=user_input
            )
            audio_parser_prompt_overview = self.lf_prompt_audio_overview.replace(
                "__bio__", bio
            )

            messages_overall = [
                {"role": "system", "content": audio_parser_prompt_overview},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_info_overall + language_desc}
                    ],
                },
            ]

            message_list = [messages_overall]
            max_retry_list = [2]

            for i in range(num_segments):
                user_info_segment = user_info.format(
                    content=content, speech=speech_segments[i], user_input=user_input
                )
                messages_segment = [
                    {"role": "system", "content": self.lf_prompt_audio_breakdown},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_info_segment + language_desc}
                        ],
                    },
                ]
                message_list.append(messages_segment)
                max_retry_list.append(2)

            results = []
            for messages in message_list:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.0,
                    max_retries=max_retries,
                    timeout=request_timeout,
                    response_format={"type": "json_object"},
                )
                results.append(response.choices[0].message.content)

            try:
                title = results[0].get("Title", "")
                overview = results[0].get("Overview", "")
                breakdown = {}
                for res_p in results[1:]:
                    breakdown = {**breakdown, **res_p.get("Breakdown", {})}
                tmpl = "{}\n{}"
                formated_breakdown = ""
                for subtitle, key_points in breakdown.items():
                    formated_breakdown += f"\n**{subtitle}**\n"
                    for key_point in key_points:
                        if len(key_point) != 3:
                            raise ValueError(
                                f"Unexpected length of key_point: {key_point}"
                            )
                        timestamps = (
                            key_point[2].replace("，", ",").replace(" ", "").split(",")
                        )
                        std_timestamps = "".join(
                            [
                                f"[_TIMESTAMP_]('{timestamp}')"
                                for timestamp in timestamps
                            ]
                        )
                        formated_breakdown += (
                            f"- **{key_point[0]}**: {key_point[1]}{std_timestamps}\n"
                        )

                insight = tmpl.format(overview, formated_breakdown)
                return insight, title
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise RuntimeError(f"Unexpected error: {e}")
        else:
            user_info = user_info.format(
                content=content, speech=speech, user_input=user_input
            )
            prompt_audio_parser = self.lf_prompt_audio_parser.replace("__bio__", bio)

            messages = [
                {"role": "system", "content": prompt_audio_parser},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_info + language_desc}],
                },
            ]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=4096,
                temperature=0.0,
                max_retries=max_retries,
                timeout=request_timeout,
                response_format={"type": "json_object"},
            )
            api_res_dict = response.choices[0].message.content

            try:
                title = api_res_dict.get("Title", "")
                overview = api_res_dict.get("Overview", "")
                breakdown = api_res_dict.get("Breakdown", {})
                tmpl = "{}\n{}"
                formated_breakdown = ""
                for subtitle, key_points in breakdown.items():
                    formated_breakdown += f"\n**{subtitle}**\n"
                    for key_point in key_points:
                        if len(key_point) != 3:
                            raise ValueError(
                                f"Unexpected length of key_point: {key_point}"
                            )
                        timestamps = (
                            key_point[2].replace("，", ",").replace(" ", "").split(",")
                        )
                        std_timestamps = "".join(
                            [
                                f"[_TIMESTAMP_]('{timestamp}')"
                                for timestamp in timestamps
                            ]
                        )
                        formated_breakdown += (
                            f"- **{key_point[0]}**: {key_point[1]}{std_timestamps}\n"
                        )

                insight = tmpl.format(overview, formated_breakdown)

                return insight, title

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise RuntimeError(f"Unexpected error: {e}")

    def _insighter_doc(
        self,
        bio: Dict[str, str],
        content: str,
        max_retries: int,
        request_timeout: int,
        file_content: Dict[str, Any],
        max_tokens: int = 3000,
        filter=equidistant_filter,
    ) -> tuple[str, str]:
        """Process document content to generate insights.
        
        Args:
            bio: Dictionary containing user biography information
            content: Text content or hint about the document
            max_retries: Maximum number of API call retries
            request_timeout: Timeout for API calls in seconds
            file_content: Dictionary containing document content
            max_tokens: Maximum tokens for generation
            filter: Function to filter document chunks
            
        Returns:
            Tuple of (insight, title) strings
        """
        user_info = """# Hint # 
                    "{hint}"

                    # Content #
                    "{content}"

                    # User Instruction #
                    "{user_input}"
                    """
        user_input = "Here are some content and their hint. Please follow the WorkFlow and do your best. Ensure that your response is in a parseable JSON format.  "
        language_desc = select_language_desc(self.preferred_language)

        segment_list = [self.lf_prompt_doc_overview, self.lf_prompt_doc_breakdown]
        messages_list = []
        max_retry_list = []
        alarm_mesg_list = []
        for i in range(len(segment_list)):
            DOC_PARSER_PROMPT = segment_list[i]
            raw_text = DOC_PARSER_PROMPT + user_input + user_info + language_desc
            upper_bound = cal_upperbound(
                model_limit=7000 + max_tokens,
                generage_limit=max_tokens,
                tolerance=500,
                raw=raw_text,
            )
            # Chunk and truncate
            chunk_size = 512
            chunk_num = upper_bound // chunk_size + 1

            if self.model_name is None:
                self.user_llm_config = self.user_llm_config_service.get_available_llm()
                self.client = OpenAI(
                    api_key=self.user_llm_config.chat_api_key,
                    base_url=self.user_llm_config.chat_endpoint,
                )
                self.model_name = self.user_llm_config.chat_model_name

            spliter = TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=0,
                model_name=self.model_name.replace("openai/", ""),
            )

            tmp = file_content.get("content", "")
            doc_content = "\n".join(tmp)
            splits = spliter.split_text(doc_content)
            use_content = chunk_filter(
                splits, filter, filtered_chunks_n=chunk_num, separator="\n", spacer="\n"
            )
            doc_content = get_safe_content_turncate(
                use_content, self.model_name.replace("openai/", ""), max_tokens=upper_bound
            )

            user_content = user_info.format(
                hint=content, content=doc_content, user_input=user_input
            )
            if "__global_bio__" in DOC_PARSER_PROMPT:
                DOC_PARSER_PROMPT = DOC_PARSER_PROMPT.replace(
                    "__about_me__", bio["about_me"]
                )
                DOC_PARSER_PROMPT = DOC_PARSER_PROMPT.replace(
                    "__global_bio__", bio["global_bio"]
                )
                DOC_PARSER_PROMPT = DOC_PARSER_PROMPT.replace(
                    "__status_bio__", bio["status_bio"]
                )

            messages = [
                {"role": "system", "content": DOC_PARSER_PROMPT},
                {"role": "user", "content": user_content + language_desc},
            ]
            messages_list.append(messages)

        results = []
        for messages in messages_list:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
                timeout=request_timeout,
                response_format={"type": "json_object"},
            )
            results.append(json.loads(response.choices[0].message.content))
        try:
            title = results[0].get("Title")
            overview = results[0].get("Overview")
            breakdown = results[1].get("Breakdown", {})

            tmpl = "{}\n{}"

            formated_breakdown = ""
            for subtitle, key_points in breakdown.items():
                formated_breakdown += f"\n**{subtitle}**\n"

                if not isinstance(key_points, list):
                    raise RuntimeError(
                        f"Unexpected generated result: {json.dumps(breakdown)}"
                    )

                for key_point in key_points:
                    if isinstance(key_point, list) and len(key_point) == 2:
                        formated_breakdown += f"- **{key_point[0]}**: {key_point[1]}\n"
                    else:
                        raise RuntimeError(
                            f"Unexpected generated result in key_points: {json.dumps(breakdown)} expected a list of length 2."
                        )

            insight = tmpl.format(overview, formated_breakdown)

            return insight, title

        except Exception as e:
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Unexpected error: {e}")

    def insighter(self, inputs: InsighterInput) -> Dict[str, str]:
        """Generate insights from document inputs.
        
        Args:
            inputs: Structured input parameters containing file and bio information
            
        Returns:
            Dictionary containing title and insight
        """
        try:
            datatype = DataType(inputs.file_info.data_type)
        except ValueError:
            logger.warning(
                "Unsupported dataType: %s. Processing as DOCUMENT by default",
                inputs.file_info.data_type,
            )
            datatype = DataType.DOCUMENT

        logger.info("input filename=%s", inputs.file_info.filename)
        logger.info(
            "input content=%s (first 100 characters)",
            inputs.file_info.content.strip()[:100],
        )

        bio = {
            "global_bio": inputs.bio_info.global_bio.split("### Conclusion ###")[
                -1
            ].strip("\n ")
            if inputs.bio_info.global_bio
            else "User has no biography right now",
            "status_bio": inputs.bio_info.status_bio.split(
                "** User Activities Overview **"
            )[-1]
            .strip("** Physical and mental health status **")[0]
            .strip("\n")
            if inputs.bio_info.status_bio
            else "",
            "about_me": inputs.bio_info.about_me.strip("\n")
            if inputs.bio_info.about_me
            else "",
        }

        text_len = len(self._tokenizer.encode(inputs.file_info.content))

        if text_len > 20 or inputs.file_info.file_content:
            if datatype == DataType.IMAGE:
                insight, title = self._insighter_image(
                    bio=bio,
                    content=inputs.file_info.content,
                    max_retries=self.max_retries_summarize,
                    request_timeout=180,
                    file_content=inputs.file_info.file_content,
                )
            elif datatype == DataType.AUDIO:
                insight, title = self._insighter_audio(
                    bio=bio,
                    content=inputs.file_info.content,
                    max_retries=self.max_retries_summarize,
                    request_timeout=180,
                    file_content=inputs.file_info.file_content,
                )
            else:
                insight, title = self._insighter_doc(
                    bio=bio,
                    content=inputs.file_info.content,
                    max_retries=self.max_retries_summarize,
                    request_timeout=180,
                    file_content=inputs.file_info.file_content,
                )
        else:
            logger.warning("less than 20 characters, use filename as title")
            title, insight = inputs.file_info.content, inputs.file_info.content
            if inputs.file_info.filename:
                logger.info("use filename as title")
                title = inputs.file_info.filename

        t1 = time.time()
        logger.warning(
            "Insighter: title=%s, summary=%s",
            title,
            insight,
        )

        return {
            "title": title,
            "insight": insight,
        }

    def __serial_summary_filter(
        self, summaries: List[str], chunks_list: List[List[str]], separator: str = "", filtered_chunks_n: int = 6
    ) -> List[str]:
        """Filter and combine summaries with relevant chunks.
        
        Args:
            summaries: List of summary strings
            chunks_list: List of lists containing text chunks
            separator: String to join chunks and summaries
            filtered_chunks_n: Maximum number of chunks to filter
            
        Returns:
            List of combined content strings
        """
        # Skip summary when chunks length is 0, otherwise combine summary with some adjacent chunks
        use_contents = []
        for summary, chunks in zip(summaries, chunks_list):
            # When chunks exceed filtered_chunks_n-1, this is not the final summarization round
            if len(chunks) > filtered_chunks_n - 1:
                use_content = separator.join([summary, *chunks[:5]])
            # When chunks are between 0 and filtered_chunks_n-1, this is the final round
            elif len(chunks) > 0:
                use_content = separator.join([summary, *chunks])
            else:
                # When chunks are 0, summary is done, skip this round to avoid using resources
                continue
            use_contents.append(use_content)
        return use_contents

    def _summarize_title_abstract_keywords(
        self,
        content: str or List[str],
        filename: str,
        file_type: str,
        request_timeout: int,
        max_retries: int,
        preferred_language: str,
        filter=equidistant_filter,
    ) -> tuple[str, str, List[str]] or List[tuple[str, str, List[str]]]:
        """Generate title, abstract and keywords from content.
        
        Args:
            content: String or list of strings to summarize
            filename: Name of the file being summarized
            file_type: Type of file (document, image, audio, etc.)
            request_timeout: Timeout for API calls in seconds
            max_retries: Maximum number of API call retries
            preferred_language: Language to use for generation
            filter: Function to filter content chunks
            
        Returns:
            Single tuple or list of tuples containing (title, summary, keywords)
        """
        upper_limit = 8192
        filtered_chunks_n = 14
        max_tokens = 512

        if isinstance(content, str):
            inputs = [content]
        else:
            inputs = content

        filename = filename or ""
        if not filename:
            filename_desc = ""
        else:
            filename_desc = f"Filename: {filename}\n"

        def get_text_generate(_requests):
            language_desc = ""
            prompt = NOTE_SUMMARY_PROMPT.replace("{language_desc}", language_desc)
            messages = [
                [
                    {"role": "user", "content": prompt.format(**_request)},
                    {
                        "role": "system",
                        "content": f"""User Preferred Language: {preferred_language}, you should use this language to generate the title, summary.
                    Don't to start the summary section with sentences like "This document", "This text" or "This article", but describe the content directly.""",
                    },
                ]
                for _request in _requests
            ]

            logger.info("generate inputs: %s", _requests)

            responses = [
                self.client.chat.completions.create(
                    model=self.model_name,
                    messages=msg,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    timeout=request_timeout,
                )
                for msg in messages
            ]

            return responses

        spliter = TokenParagraphSplitter(chunk_size=512, chunk_overlap=0)
        if filter is self.__serial_summary_filter:
            # Serial fine-grained full-text summary
            chunks_list = [spliter.split_text(each) for each in inputs]
            # Maximum number of summaries needed [K summaries can handle docs with 5K+1 chunks]
            max_summary_times = int(
                (max([len(chunks) for chunks in chunks_list]) + 4) / 5
            )
            results = [() for i in range(len(inputs))]
            # Initialize summaries with first chunk content
            # Set to empty string if chunks length is 0
            summaries = [chunks[0] if len(chunks) > 0 else "" for chunks in chunks_list]
            # When chunks length is 1, set to [""], requires one summary
            # When chunks length is 0, set to empty list, no summary needed
            chunks_list = [
                [] if len(chunks) == 0 else ([""] if len(chunks) == 1 else chunks[1:])
                for chunks in chunks_list
            ]
            for i in range(max_summary_times):
                use_contents = self.__serial_summary_filter(summaries, chunks_list)
                requests = [
                    {
                        "content": use_content,
                        "file_type": file_type,
                        "filename_desc": filename_desc,
                    }
                    for use_content in use_contents
                ]
                responses = get_text_generate(requests)
                tmp_results = get_summarize_title_keywords(responses)
                for doc_id, chunks in enumerate(chunks_list):
                    index = 0
                    # Documents participating in this round of summaries
                    if len(chunks) > 0:
                        # Update result (title, abstract, keywords)
                        results[doc_id] = tmp_results[index]
                        # Update summary list
                        summaries[doc_id] = tmp_results[index][1]
                        # Update chunks list to be summarized
                        chunks_list[doc_id] = chunks_list[doc_id][5:]
                        index += 1
        else:
            requests = []
            for each in inputs:
                splits = spliter.split_text(each)
            # Sampling-based full text summary approach
            # Keep beginning and end, can skip middle. End is useful for company signatures and information, reducing model hallucination
            # Also keep one extra chunk at the end to avoid issues with short final chunks providing insufficient information
            use_content = chunk_filter(
                splits,
                filter,
                filtered_chunks_n=filtered_chunks_n,
                separator="\n",
                spacer="\n……\n……\n……\n",
            )
            if self.model_name is None:
                self.user_llm_config = self.user_llm_config_service.get_available_llm()
                self.client = OpenAI(
                    api_key=self.user_llm_config.chat_api_key,
                    base_url=self.user_llm_config.chat_endpoint,
                )
                self.model_name = self.user_llm_config.chat_model_name

            requests.append(
                {
                    "content": get_safe_content_turncate(
                        use_content,
                        self.model_name.replace("openai/", ""),
                        max_tokens=upper_limit,
                    ),
                    "file_type": file_type,
                    "filename_desc": filename_desc,
                }
            )
            responses = get_text_generate(requests)
            results = get_summarize_title_keywords(responses)

        logger.debug("results: %s", results)
        if isinstance(content, str):
            return results[0]
        else:
            return results

    def summarizer(self, inputs: SummarizerInput) -> Dict[str, Any]:
        """Generate summary from document inputs.
        
        Args:
            inputs: Structured input parameters containing file information and insight
            
        Returns:
            Dictionary containing title, summary and keywords
        """
        bottom_summary_len = 200

        datatype = inputs.file_info.data_type
        filename = inputs.file_info.filename
        md = inputs.file_info.content  # hint

        inner_content = inputs.file_info.file_content.get("content")
        insight = inputs.insight

        md = md + "\n" + inner_content

        md = f"insight: {insight}\ncontent: {md}"

        try:
            datatype = DataType(datatype)
        except ValueError:
            logger.warning("Unsupported dataType: %s. Processing as DOCUMENT by default", datatype)
            datatype = DataType.DOCUMENT

        logger.info("input filename=%s", filename)
        logger.info("input content=%s (first 100 characters)", md.strip()[:100])
        t0 = time.time()
        bottom_summary = self._tokenizer.decode(
            self._tokenizer.encode(insight)[:bottom_summary_len]
        )

        if len(self._tokenizer.encode(md)) > 20:
            title, summary, keywords = self._summarize_title_abstract_keywords(
                md,
                filename=filename,
                file_type=datatype.value,
                request_timeout=self.timeout_summarize,
                max_retries=self.max_retries_summarize,
                preferred_language=self.preferred_language,
            )
            if not (title or summary or keywords):
                logger.warning("summary failed, use insight as summary")
                title, summary, keywords = filename, bottom_summary, []
                if filename:
                    title = filename
        else:
            logger.warning("less than 20 characters, use filename as title")
            title, summary, keywords = md, md, []
            if filename:
                title = filename

        t1 = time.time()
        logger.warning(
            "MarkdownChunkAPI summarize_title_abstract_keywords(): time spent %.2f seconds, title=%s, summary=%s",
            t1 - t0,
            title,
            summary,
        )

        return {"title": title, "summary": summary, "keywords": keywords}
