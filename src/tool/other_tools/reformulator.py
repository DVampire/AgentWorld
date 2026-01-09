"""Reformulator tool for reformulating final answers from agent conversations."""
from typing import List, Dict, Any, Optional
from pydantic import Field, BaseModel

from src.tool.types import Tool, ToolResponse, ToolExtra
from src.message.types import SystemMessage, HumanMessage
from src.model import model_manager
from src.logger import logger
from src.utils import dedent
from src.registry import TOOL


_REFORMULATOR_TOOL_DESCRIPTION = """Reformulator tool for reformulating final answers from agent conversations.
This tool takes the original task and the conversation history, then uses an LLM to extract and format the final answer.
Use this tool when you need to produce a clean, formatted final answer from a conversation transcript.

Args:
- task (str): The original task/question that was asked
- data (List[str]): Conversation history in the form of a list of message texts.

Example: {"name": "reformulator", "args": {"task": "What is the capital of France?", "data": ["The capital of France is Paris.", "The capital of France is Paris."]}}.
"""


class ReformulatedAnswer(BaseModel):
    """Response format for reformulated final answer."""
    final_answer: str = Field(description="The final answer extracted from the conversation. Should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. Must adhere to any formatting instructions specified in the original question.")

@TOOL.register_module(force=True)
class ReformulatorTool(Tool):
    """A tool for reformulating final answers from agent conversations."""
    
    name: str = "reformulator"
    description: str = _REFORMULATOR_TOOL_DESCRIPTION
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the tool")
    require_grad: bool = Field(default=False, description="Whether the tool requires gradients")
    
    model_name: str = Field(default="openrouter/gemini-3-flash-preview", description="The model to use for reformulation.")
    
    def __init__(self, model_name: Optional[str] = None, require_grad: bool = False, **kwargs):
        """
        Initialize the reformulator tool.
        
        Args:
            model_name: The model to use for reformulation. Default: "openrouter/gemini-3-flash-preview"
        """
        super().__init__(require_grad=require_grad, **kwargs)
        if model_name:
            self.model_name = model_name
    
    async def __call__(
        self, 
        task: str, 
        data: List[str],
        **kwargs
    ) -> ToolResponse:
        """
        Reformulate the final answer from a conversation transcript.
        
        Args:
            task: The original task/question that was asked
            data: Conversation history in the form of a list of message texts.
        Returns:
            ToolResponse with the reformulated final answer
        """
        try:
            
            # Build system message
            system_prompt = dedent(f"""
                You are a helpful assistant that reformulates the final answer from a conversation transcript.
            """)
            
            data_string = '\n'.join(data)
            
            agent_message_prompt = dedent(f"""
                Original task:
                {task}
                                                        
                Conversation history:
                {data_string}
                
                Extract and format the final answer from the conversation history above.
                
                Instructions for formatting the final answer:
                - Your FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
                - ADDITIONALLY, your FINAL ANSWER MUST adhere to any formatting instructions specified in the original question (e.g., alphabetization, sequencing, units, rounding, decimal places, etc.)
                - You MUST pay attention to the required units of the calculation result. For example, if the question asks "how many thousand hours...", then the answer `1000 hours` should be `1`, not `1000`.
                - You MUST pay attention to extracting key stage names, personal names, and location names when the task required.
                - If you are asked for a number, express it numerically (i.e., with digits rather than words), don't use commas, and DO NOT INCLUDE UNITS such as $ or USD or percent signs unless specified otherwise.
                - If you are asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.
                - If you are asked for a comma separated list, apply the above rules depending on whether the elements are numbers or strings.
                - If you are unable to determine the final answer, output 'Unable to determine'
            
                Reformulated answer is:
            """)
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=agent_message_prompt)
            ]
            
            # Call model with response_format
            response = await model_manager(
                model=self.model_name, 
                messages=messages,
                response_format=ReformulatedAnswer
            )
            
            if not response.success:
                return ToolResponse(
                    success=False,
                    message=f"Failed to reformulate answer: {response.message}"
                )
            
            # Extract final answer from structured response
            if response.extra and response.extra.parsed_model:
                reformulated_answer = response.extra.parsed_model
                final_answer = reformulated_answer.final_answer
                logger.info(f"> Reformulated answer: {final_answer}")
                
                message = final_answer
                return ToolResponse(success=True, 
                                    message=message, 
                                    extra=ToolExtra(
                                        data={"original_response": response.message},
                                        parsed_model=reformulated_answer
                                    )
                                )
            else:
                # Fallback: parse from text response
                response_text = str(response.message)
                if "FINAL ANSWER: " in response_text:
                    final_answer = response_text.split("FINAL ANSWER: ")[-1].strip()
                else:
                    final_answer = response_text.strip()
                
                logger.info(f"> Reformulated answer (fallback): {final_answer}")
                
                message = final_answer
                
                return ToolResponse(success=True, 
                                    message=message, 
                                    extra=ToolExtra(
                                        data={"original_response": response_text}
                                    )
                                )
            
        except Exception as e:
            logger.error(f"Error in reformulator tool: {e}")
            return ToolResponse(
                success=False,
                message=f"Error reformulating answer: {str(e)}"
            )

