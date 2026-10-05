from typing import List, Dict, Any, Type, Optional

from lpm_kernel.api.domains.kernel2.dto.chat_dto import ChatRequest
from lpm_kernel.api.domains.kernel2.services.prompt_builder import (
    SystemPromptBuilder,
    SystemPromptStrategy,
    BasePromptStrategy,
    RoleBasedStrategy,
    KnowledgeEnhancedStrategy,
)

class MessageBuilder:
    """Base class for building chat messages"""
    
    def build_messages(self, context: Optional[Any] = None) -> List[Dict[str, Any]]:
        """Build messages for chat completion"""
        raise NotImplementedError()


class MultiTurnMessageBuilder(MessageBuilder):
    """Message builder for multi-turn chat"""
    
    def __init__(self, chat_request: ChatRequest, strategy_chain: List[Type[SystemPromptStrategy]] = None):
        """
        Initialize the builder with a chat request and optional strategy chain.
        
        Args:
            chat_request: The chat request to build messages for
            strategy_chain: List of strategy classes in the order they should be applied.
                          Default is [RoleBasedStrategy, KnowledgeEnhancedStrategy]
        """
        self.chat_request = chat_request
        self.strategy_chain = strategy_chain if strategy_chain is not None else [
            BasePromptStrategy, RoleBasedStrategy, KnowledgeEnhancedStrategy
        ]

    def build_messages(self, context: Optional[Any] = None) -> List[Dict[str, Any]]:
        """Place one system prompt before copied conversation messages."""
        current_strategy = None
        for strategy_class in self.strategy_chain:
            current_strategy = (
                strategy_class() if current_strategy is None
                else strategy_class(base_strategy=current_strategy)
            )
        if current_strategy is None:
            raise ValueError("No strategy provided")
        builder = SystemPromptBuilder()
        builder.set_strategy(current_strategy)
        system_prompt = builder.build_prompt(self.chat_request, context)
        messages = [dict(message) for message in self.chat_request.messages
                    if message.get("role") != "system"]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        return messages
