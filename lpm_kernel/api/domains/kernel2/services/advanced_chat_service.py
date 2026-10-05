"""
Service for handling advanced chat mode with multiple phases
"""

import json
import logging
from typing import Dict, Any, Union, Iterator

from lpm_kernel.api.services.expert_llm_service import expert_llm_service
from lpm_kernel.api.domains.kernel2.dto.advanced_chat_dto import (
    AdvancedChatRequest,
    ValidationResult,
    AdvancedChatResponse
)
from lpm_kernel.api.domains.kernel2.dto.chat_dto import ChatRequest
from lpm_kernel.api.domains.kernel2.services.chat_service import chat_service
from lpm_kernel.api.domains.kernel2.services.advanced_prompt_strategies import (
    RequirementEnhancementStrategy,
    ExpertSolutionStrategy,
    SolutionValidatorStrategy,
    SolutionFormatterStrategy,
)
from lpm_kernel.api.domains.kernel2.services.prompt_builder import BasePromptStrategy

logger = logging.getLogger(__name__)

class AdvancedChatService:
    """Service for handling advanced chat mode"""

    def enhance_requirement(self, request: AdvancedChatRequest) -> str:
        """Enhance the requirement with knowledge context"""
        logger.info("Starting requirement enhancement phase...")
        
        # Convert AdvancedChatRequest to ChatRequest
        chat_request = ChatRequest(
            messages=[{"role": "user", "content": request.requirement}],
            metadata={"enable_l0_retrieval": request.enable_l0_retrieval,
                      "enable_l1_retrieval": request.enable_l1_retrieval},
            temperature=request.temperature
        )
        logger.info(f"Created chat request with message: {request.requirement[:100]}...")
        
        # Use chat service with RequirementEnhancementStrategy
        logger.info("Calling chat service with RequirementEnhancementStrategy...")
        response = chat_service.chat(
            request=chat_request,
            strategy_chain=[BasePromptStrategy, RequirementEnhancementStrategy],
            stream=False,
            json_response=False
        )
        
        enhanced_requirement = response.choices[0].message.content
        logger.info(f"Requirement enhancement completed. Result: {enhanced_requirement[:100]}...")
        return enhanced_requirement

    def generate_solution(self, requirement: str, temperature: float) -> str:
        """Generate solution based on enhanced requirement"""
        logger.info("Starting solution generation phase with expert model...")
        logger.info(f"Input requirement: {requirement[:100]}...")
        
        chat_request = ChatRequest(
            messages=[{"role": "user", "content": requirement}],
            metadata={"enable_l0_retrieval": False, "enable_l1_retrieval": False},
            temperature=temperature
        )
        
        logger.info("Calling chat service with ExpertSolutionStrategy using expert model...")
        response = chat_service.chat(
            request=chat_request,
            strategy_chain=[BasePromptStrategy, ExpertSolutionStrategy],
            stream=False,
            json_response=False,
            client=expert_llm_service.client  # Use expert model
        )
        
        solution = response.choices[0].message.content
        logger.info(f"Solution generation completed. Result: {solution[:100]}...")
        return solution

    def validate_solution(self, requirement: str, solution: str) -> ValidationResult:
        """Validate if solution meets requirements"""
        logger.info("Starting solution validation phase...")
        logger.info(f"Validating solution of length {len(solution)} characters...")
        
        chat_request = ChatRequest(
            messages=[{"role": "user", "content": f"""
            Requirement:
            {requirement}
            
            Solution:
            {solution}
            """}],
            metadata={"enable_l0_retrieval": False, "enable_l1_retrieval": False},
            temperature=0.2  # Lower temperature for more consistent validation
        )
        
        logger.info("Calling chat service with SolutionValidatorStrategy...")
        response = chat_service.chat(
            request=chat_request,
            strategy_chain=[BasePromptStrategy, SolutionValidatorStrategy],
            stream=False,
            json_response=False
        )
        
        validation_text = response.choices[0].message.content
        try:
            validation_dict = json.loads(validation_text)
            validation_result = ValidationResult(**validation_dict)
            logger.info(f"Validation completed. Result: {validation_result}")
            return validation_result
        except Exception as e:
            logger.error(f"Failed to parse validation result: {str(e)}")
            logger.error(f"Raw validation text: {validation_text}")
            return ValidationResult(is_valid=False, feedback="Failed to validate solution")

    def format_solution(self, solution: str) -> str:
        """Format the final solution"""
        logger.info("Starting solution formatting phase...")
        logger.info(f"Formatting solution of length {len(solution)} characters...")
        
        chat_request = ChatRequest(
            messages=[{"role": "user", "content": solution}],
            metadata={"enable_l0_retrieval": False, "enable_l1_retrieval": False},
            temperature=0.3  # Lower temperature for more consistent formatting
        )
        
        logger.info("Calling chat service with SolutionFormatterStrategy...")
        response = chat_service.chat(
            request=chat_request,
            strategy_chain=[BasePromptStrategy, SolutionFormatterStrategy],
            stream=False,
            json_response=False
        )
        
        formatted_solution = response.choices[0].message.content
        logger.info(f"Formatting completed. Result length: {len(formatted_solution)} characters")
        logger.info(f"First 100 characters of formatted solution: {formatted_solution[:100]}...")
        return formatted_solution

    def format_final_response(self, solution: str, stream: bool = True) -> Union[Iterator[Dict[str, Any]], Dict[str, Any]]:
        """Format and stream the final response using base model"""
        logger.info("Formatting final response with base model...")
        
        # build system prompt to instruct the base model to express
        system_prompt = """You are a helpful AI assistant. Your task is to express the given solution in a clear, 
        natural, and engaging way. Follow these guidelines:
        1. Maintain the technical accuracy of the solution
        2. Use a conversational but professional tone
        3. Break down complex concepts into digestible parts
        4. Highlight key points and important considerations
        5. Add relevant examples or analogies when helpful
        
        The solution will be provided in the user's message. Respond as if you are directly 
        explaining the solution to the user."""
        
        chat_request = ChatRequest(
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": solution}],
            metadata={"enable_l0_retrieval": False, "enable_l1_retrieval": False},
            temperature=0.3  # use lower temp to keep stability
        )
        
        logger.info("Streaming final response...")
        return chat_service.chat(
            request=chat_request,
            stream=stream,
            json_response=False
        )

    def process_advanced_chat(self, request: AdvancedChatRequest) -> AdvancedChatResponse:
        """Process advanced chat request through all phases"""
        logger.info(f"Starting advanced chat processing with max_iterations={request.max_iterations}...")
        
        # 1. Enhance requirement
        logger.info("Phase 1: Requirement Enhancement")
        enhanced_requirement = self.enhance_requirement(request)
        
        # 2. Generate initial solution
        logger.info("Phase 2: Initial Solution Generation")
        current_solution = self.generate_solution(enhanced_requirement, request.temperature)
        
        # 3. Validation and refinement loop
        logger.info("Phase 3: Validation and Refinement Loop")
        validation_history = []
        final_format = None
        
        for iteration in range(request.max_iterations):
            logger.info(f"Starting iteration {iteration + 1}/{request.max_iterations}")
            
            # Validate current solution
            validation_result = self.validate_solution(enhanced_requirement, current_solution)
            validation_history.append(validation_result)
            
            if validation_result.is_valid:
                logger.info("Solution validated successfully")
                # Format solution if valid
                logger.info("Phase 4: Final Formatting")
                final_format = self.format_solution(current_solution)
                break
            elif iteration < request.max_iterations - 1:
                logger.info(f"Solution needs improvement. Feedback: {validation_result.feedback}")
                # Generate improved solution based on feedback
                current_solution = self.generate_solution(
                    f"{enhanced_requirement}\n\nPrevious attempt feedback: {validation_result.feedback}",
                    request.temperature
                )
        
        # use base model to construct the final response
        final_response = self.format_final_response(final_format or current_solution, stream=True)
        
        logger.info("Advanced chat processing completed")
        return AdvancedChatResponse(
            enhanced_requirement=enhanced_requirement,
            solution=current_solution,
            validation_history=validation_history,
            final_format=final_format,
            final_response=final_response
        )


# Global instance
advanced_chat_service = AdvancedChatService()
