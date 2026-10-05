"""
Advanced Talk API Routes

This module provides enhanced chat capabilities with features like:
- Multi-turn conversations
- System prompt management
- Conversation history tracking
- Streaming response
- Advanced mode with multi-phase processing
"""

import logging
from typing import Iterator
from flask import Blueprint, jsonify
from flask_pydantic import validate

from lpm_kernel.api.common.responses import APIResponse
from lpm_kernel.api.services.local_llm_service import local_llm_service
from lpm_kernel.api.domains.kernel2.dto.chat_dto import ChatRequest
from lpm_kernel.api.domains.kernel2.dto.advanced_chat_dto import AdvancedChatRequest
from lpm_kernel.api.domains.kernel2.services.chat_service import chat_service
from lpm_kernel.api.domains.kernel2.services.advanced_chat_service import advanced_chat_service

logger = logging.getLogger(__name__)

talk_bp = Blueprint('talk', __name__, url_prefix='/api/talk')

@talk_bp.route("/chat", methods=["POST"])
@validate()
def chat(body: ChatRequest):
    """
    Chat endpoint - streaming response
    
    Request: ChatRequest JSON object containing:
    - messages: OpenAI-compatible system/user/assistant conversation
    - metadata.role_id: optional role UUID
    - metadata.enable_l0_retrieval: bool, personal chat default true
    - metadata.enable_l1_retrieval: bool, personal chat default false
    - temperature: float, temperature parameter for randomness, default 0.01
    - max_tokens: int, maximum tokens to generate, default 2000
    """
    try:
        # 1. Check server status
        status = local_llm_service.get_server_status()
        if not status.is_running:
            error_response = APIResponse.error("LLama server is not running")
            return local_llm_service.handle_stream_response(iter([{"error": error_response}]))

        try:
            # 2. Use chat service to handle request
            response = chat_service.chat(
                request=body,
                stream=True,
                json_response=False,
            )
            return local_llm_service.handle_stream_response(response)

        except Exception as e:
            logger.error(f"API call failed: {str(e)}", exc_info=True)
            error_response = APIResponse.error(f"API call failed: {str(e)}")
            return local_llm_service.handle_stream_response(iter([{"error": error_response}]))

    except Exception as e:
        logger.error(f"Request processing failed: {str(e)}", exc_info=True)
        error_response = APIResponse.error(f"Request processing failed: {str(e)}")
        return local_llm_service.handle_stream_response(iter([{"error": error_response}]))


@talk_bp.route("/chat_json", methods=["POST"])
@validate()
def chat_json(body: ChatRequest):
    """
    Chat endpoint - JSON response (non-streaming)
    
    Used for testing if the model supports JSON structure responses.
    
    Request: ChatRequest JSON object, same as chat endpoint
    
    Response:
    JSON object containing:
    - id: str, unique response identifier
    - object: str, object type, usually "chat.completion"
    - created: int, creation timestamp
    - model: str, model name used
    - system_fingerprint: str, system fingerprint
    - choices: List[Dict], containing generated content, each choice contains:
        - index: int, choice index
        - message: Dict, containing generated content
            - role: str, role (usually "assistant")
            - content: str, generated text content
            - function_call: Optional[Dict], if there's a function call
    """
    try:
        # 1. Check server status
        status = local_llm_service.get_server_status()
        if not status.is_running:
            return jsonify(APIResponse.error("LLama server is not running"))

        try:
            # 2. Use chat service to handle request
            response = chat_service.chat(
                request=body,
                stream=False,
                json_response=True,
            )
            return jsonify(APIResponse.success(response))

        except Exception as e:
            logger.error(f"API call failed: {str(e)}", exc_info=True)
            return jsonify(APIResponse.error(f"API call failed: {str(e)}"))

    except Exception as e:
        logger.error(f"Request processing failed: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Request processing failed: {str(e)}"))


@talk_bp.route("/advanced_chat", methods=["POST"])
@validate()
def advanced_chat(body: AdvancedChatRequest):
    """
    Advanced chat endpoint - multi-phase processing
    
    This endpoint implements a sophisticated chat process with multiple phases:
    1. Requirement Enhancement - Enhances user's rough requirement with context
    2. Expert Solution - Generates solution based on enhanced requirement
    3. Validation and Refinement - Validates and improves solution iteratively
    
    Request: AdvancedChatRequest JSON object containing:
    - requirement: str, user's rough requirement
    - max_iterations: int, maximum number of refinement iterations (default: 3)
    - temperature: float, temperature for model generation (default: 0.01)
    - enable_l0_retrieval: bool, whether to enable L0 knowledge retrieval (default: true)
    - enable_l1_retrieval: bool, whether to enable L1 knowledge retrieval (default: true)
    
    Response:
    JSON object containing:
    - enhanced_requirement: str, enhanced requirement with context
    - solution: str, generated solution
    - validation_history: List[ValidationResult], history of validation results
    - final_format: Optional[str], final formatted solution if valid
    """
    try:
        # 1. Check server status
        status = local_llm_service.get_server_status()
        if not status.is_running:
            return jsonify(APIResponse.error("LLama server is not running"))

        try:
            # 2. Process advanced chat request
            response = advanced_chat_service.process_advanced_chat(body)
            
            # process streaming responses
            if isinstance(response.final_response, Iterator):
                return local_llm_service.handle_stream_response(response.final_response)
                
            # if not streaming, return the final response
            return jsonify({
                "enhanced_requirement": response.enhanced_requirement,
                "solution": response.solution,
                "validation_history": [v.dict() for v in response.validation_history],
                "final_format": response.final_format,
                "final_response": response.final_response
            })
            
        except Exception as e:
            logger.error(f"Advanced chat processing failed: {str(e)}", exc_info=True)
            return jsonify(APIResponse.error(f"Advanced chat processing failed: {str(e)}"))

    except Exception as e:
        logger.error(f"Request processing failed: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Request processing failed: {str(e)}"))
