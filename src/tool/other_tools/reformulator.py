"""Reformulator tool for reformulating final answers from agent conversations."""
from typing import List, Dict, Any, Optional
from pydantic import Field

from src.tool.types import Tool, ToolResponse
from src.message.types import SystemMessage, HumanMessage
from src.model import model_manager
from src.logger import logger
from src.utils import dedent


_REFORMULATOR_TOOL_DESCRIPTION = """Reformulator tool for reformulating final answers from agent conversations.
This tool takes the original task and the conversation history, then uses an LLM to extract and format the final answer.
Use this tool when you need to produce a clean, formatted final answer from a conversation transcript.
"""

class ReformulatorTool(Tool):
    """A tool for reformulating final answers from agent conversations."""
    
    name: str = "reformulator"
    description: str = _REFORMULATOR_TOOL_DESCRIPTION
    enabled: bool = True
    
    model_name: str = Field(default="gpt-4.1", description="The model to use for reformulation.")
    
    def __init__(self, model_name: Optional[str] = None, **kwargs):
        """
        Initialize the reformulator tool.
        
        Args:
            model_name: The model to use for reformulation. Default: "gpt-4.1"
        """
        super().__init__(**kwargs)
        if model_name:
            self.model_name = model_name
    
    async def __call__(
        self, 
        task: str, 
        messages: List[Dict[str, Any]], 
        **kwargs
    ) -> ToolResponse:
        """
        Reformulate the final answer from a conversation transcript.
        
        Args:
            original_task: The original task/question that was asked
            inner_messages: List of messages from the conversation (can be dict or Message objects)
            reformulation_model: Model name to use for reformulation. If not provided, uses self.model_name
            
        Returns:
            ToolResponse with the reformulated final answer
        """
        try:
            # Build system message
            system_content = dedent(f"""
                Earlier you were asked the following:
                {task}
                Your team then worked diligently to address that request. 
                Read below a transcript of that conversation:
            """)
            
            messages = [SystemMessage(content=system_content)] + messages
                
            # Add final prompt for reformulation
            final_prompt = dedent(f"""
                Read the above conversation and output a FINAL ANSWER to the question. The question is repeated here for convenience:
                {task}
                To output the final answer, use the following template: FINAL ANSWER: [YOUR FINAL ANSWER]
                Your FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
                ADDITIONALLY, your FINAL ANSWER MUST adhere to any formatting instructions specified in the original question (e.g., alphabetization, sequencing, units, rounding, decimal places, etc.)
                * You MUST pay attention to the required units of the calculation result. For example, if the question asks "how many thousand hours...", then the answer `1000 hours` should be `1`, not `1000`.
                * You MUST pay attention to extracting key stage names, personal names, and location names when the task required.
                * If you are asked for a number, express it numerically (i.e., with digits rather than words), don't use commas, and DO NOT INCLUDE UNITS such as $ or USD or percent signs unless specified otherwise.
                * If you are asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.
                * If you are asked for a comma separated list, apply the above rules depending on whether the elements are numbers or strings.
                * If you are unable to determine the final answer, output 'FINAL ANSWER: Unable to determine'
            """)
            
            messages.append(HumanMessage(content=final_prompt))
            
            # Call model
            response = await model_manager(model=self.model_name, messages=messages)
            
            if not response.success:
                return ToolResponse(
                    success=False,
                    message=f"Failed to reformulate answer: {response.message}"
                )
            
            # Extract final answer from response
            response_text = str(response.message)
            if "FINAL ANSWER: " in response_text:
                final_answer = response_text.split("FINAL ANSWER: ")[-1].strip()
            else:
                # Fallback: use the entire response
                final_answer = response_text.strip()
            
            logger.info(f"> Reformulated answer: {final_answer}")
            
            return ToolResponse(
                success=True,
                message=final_answer,
                extra={"original_response": response_text}
            )
            
        except Exception as e:
            logger.error(f"Error in reformulator tool: {e}")
            return ToolResponse(
                success=False,
                message=f"Error reformulating answer: {str(e)}"
            )

