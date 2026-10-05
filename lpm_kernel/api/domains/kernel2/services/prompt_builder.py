"""
System prompt builder and related strategies
"""
from typing import Optional, Any
import logging

from lpm_kernel.api.domains.kernel2.dto.chat_dto import ChatRequest
from lpm_kernel.api.domains.kernel2.services.role_service import role_service
from lpm_kernel.api.domains.kernel2.services.knowledge_service import (
    default_retriever,
    default_l1_retriever,
)
from lpm_kernel.L2.training_prompt import CONTEXT_PROMPT, JUDGE_PROMPT
from lpm_kernel.L2.memory_prompt import (
    DEFAULT_MAX_MEMORY_TOKENS,
    approximate_token_count,
    build_memory_system_prompt,
)


logger = logging.getLogger(__name__)


class SystemPromptStrategy:
    """Base class for system prompt building strategies"""
    def build_prompt(self, request: ChatRequest, context: Optional[Any] = None) -> str:
        """Build system prompt"""
        raise NotImplementedError()


class BasePromptStrategy(SystemPromptStrategy):
    """Most basic system prompt building strategy"""
    def build_prompt(self, request: ChatRequest, context: Optional[Any] = None) -> str:
        """Return the basic system prompt"""
        return "\n\n".join(message.get("content", "") for message in request.messages
                            if message.get("role") == "system")


class ContextEnhancedStrategy(SystemPromptStrategy):
    """Context-enhanced system prompt building strategy"""
    def build_prompt(self, request: ChatRequest, context: Optional[Any] = None) -> str:
        """Build context-enhanced system prompt"""
        base_prompt = CONTEXT_PROMPT
        return base_prompt

class ContextCriticStrategy(SystemPromptStrategy):
    """Context-critic system prompt building strategy"""
    def build_prompt(self, request: ChatRequest, context: Optional[Any] = None) -> str:
        """Build context-critic system prompt"""
        base_prompt = JUDGE_PROMPT
        return base_prompt

class RoleBasedStrategy(SystemPromptStrategy):
    """Role-based system prompt building strategy"""
    def __init__(self, base_strategy: SystemPromptStrategy):
        self.base_strategy = base_strategy

    def build_prompt(self, request: ChatRequest, context: Optional[Any] = None) -> str:
        """Build system prompt based on role"""
        # Get role_id from metadata if available
        role_id = None
        if hasattr(request, 'metadata') and request.metadata:
            role_id = request.metadata.get('role_id')
        
        if role_id:
            role = role_service.get_role_by_uuid(role_id)
            if role:
                prompt = role.system_prompt
                return prompt
                
        prompt = self.base_strategy.build_prompt(request, context)
        # logger.info(f"RoleBasedStrategy (from base): {prompt}")
        return prompt


class KnowledgeEnhancedStrategy(SystemPromptStrategy):
    """Knowledge-enhanced system prompt building strategy"""
    def __init__(self, base_strategy: SystemPromptStrategy):
        self.base_strategy = base_strategy

    def get_user_message(self, request: ChatRequest) -> str:
        """
        Get the last user message from messages field.
        """
        if request.messages:
            # Find the last message with role='user'
            for message in reversed(request.messages):
                if message.get('role') == 'user':
                    return message.get('content', '')
        
        return ''

    def build_prompt(self, request: ChatRequest, context: Optional[Any] = None) -> str:
        """Build knowledge-enhanced system prompt"""
        base_prompt = self.base_strategy.build_prompt(request, context)
        
        metadata = request.metadata or {}
        role_id = metadata.get("role_id")
        role = role_service.get_role_by_uuid(role_id) if role_id else None
        # Explicit request values override role defaults; personal chat uses L0.
        enable_l0 = metadata.get("enable_l0_retrieval", role.enable_l0_retrieval if role else True)
        enable_l1 = metadata.get("enable_l1_retrieval", role.enable_l1_retrieval if role else False)
        if not enable_l0 and not enable_l1:
            return base_prompt
        references = []
        user_message = self.get_user_message(request)
        if user_message and enable_l0:
            knowledge = default_retriever.retrieve(user_message)
            if knowledge:
                references.append({"source": "L0 documents (current facts)", "content": knowledge})
        if user_message and enable_l1:
            knowledge = default_l1_retriever.retrieve(user_message)
            if knowledge:
                references.append({"source": "L1 summaries (may be older)", "content": knowledge})
        # Same counter and budget as training so the model sees equal reference lengths.
        return build_memory_system_prompt(base_prompt, references,
                                          token_counter=approximate_token_count,
                                          max_memory_tokens=DEFAULT_MAX_MEMORY_TOKENS)


class SystemPromptBuilder:
    """System prompt builder"""
    def __init__(self):
        self.strategy: Optional[SystemPromptStrategy] = None

    def set_strategy(self, strategy: SystemPromptStrategy):
        self.strategy = strategy

    def build_prompt(self, request: ChatRequest, context: Optional[Any] = None) -> str:
        if not self.strategy:
            raise ValueError("No strategy set for SystemPromptBuilder")
        prompt = self.strategy.build_prompt(request, context)
        # logger.info(f"Final system prompt: {prompt}")
        return prompt
