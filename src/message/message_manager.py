from typing import List, Dict, Any, Union, Optional, Type

from pydantic import BaseModel

from src.message.types import Message, HumanMessage, AIMessage, SystemMessage
from src.model.types import ModelConfig
from src.logger import logger


class MessageManager:
    """
    Render user/assistant/system messages into the schema expected by the target API
    (chat/completions vs responses), using ModelConfig provided at call time.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    async def initialize(self):
        logger.info(f"| Message manager initialized successfully.")

    def _format_role(self, message: Message) -> str:
        if isinstance(message, HumanMessage):
            return "user"
        if isinstance(message, AIMessage):
            return "assistant"
        if isinstance(message, SystemMessage):
            return "system"
        return "user"

    def _format_content_chat(self, message: Message) -> List[Dict[str, Any]]:
        if isinstance(message.content, str):
            return [{"type": "text", "text": message.content}]

        if isinstance(message.content, list):
            # Messages are already in the correct format, pass through as-is
            return message.content

        return message.content if isinstance(message.content, list) else [{"type": "text", "text": str(message.content)}]

    def _format_content_responses(self, message: Message) -> List[Dict[str, Any]]:
        if isinstance(message.content, str):
            return [{"type": "input_text", "text": message.content}]

        if isinstance(message.content, list):
            # Messages are already in the correct format, pass through as-is
            return message.content

        return message.content if isinstance(message.content, list) else [{"type": "input_text", "text": str(message.content)}]

    def _format_response_format(
        self, 
        response_format: Union[Type[BaseModel], BaseModel]
    ) -> Dict[str, Any]:
        """
        Extract parameter types, descriptions, and default values from BaseModel
        and organize them into a response format dictionary compatible with OpenAI's strict mode.
        
        This function extracts:
        - Parameter types: From the field type annotations
        - Parameter descriptions: From field docstrings or Field(description=...)
        - Parameter default values: From field defaults or model instance values
        
        Args:
            response_format: BaseModel class or instance
            
        Returns:
            Dictionary containing response format configuration with:
            - type: "json_schema"
            - json_schema: Contains name, strict mode, and schema with:
              - properties: Field types, descriptions, and default values
              - required: List of required fields
              - additionalProperties: false (required for OpenAI strict mode)
              - Additional schema definitions if present
        """
        # Get the BaseModel class if it's an instance
        if isinstance(response_format, BaseModel) and not isinstance(response_format, type):
            model_class = type(response_format)
            model_instance = response_format
        else:
            model_class = response_format
            model_instance = None
        
        # Get JSON schema from Pydantic model (includes types, descriptions, defaults)
        schema = model_class.model_json_schema()
        
        # Build a lookup for $defs to resolve references
        defs_lookup = schema.get("$defs", {})
        
        def optimize_schema(obj: Any, defs: Dict[str, Any] = None) -> Any:
            """
            Recursively process schema to:
            1. Resolve $ref references
            2. Add additionalProperties: false to all object types (OpenAI requirement)
            3. Preserve types, descriptions, and default values
            """
            if defs is None:
                defs = defs_lookup
            
            if isinstance(obj, dict):
                optimized = {}
                
                # Handle $ref references
                if "$ref" in obj:
                    ref_path = obj["$ref"]
                    if ref_path.startswith("#/$defs/"):
                        def_name = ref_path.split("/")[-1]
                        if def_name in defs:
                            # Resolve the reference and recursively optimize
                            return optimize_schema(defs[def_name], defs)
                
                # Process all keys
                for key, value in obj.items():
                    # Skip $ref as we handle it above
                    if key == "$ref":
                        continue
                    
                    # Recursively process nested structures
                    if key in ["properties", "items"]:
                        optimized[key] = optimize_schema(value, defs)
                    elif key == "anyOf" or key == "oneOf" or key == "allOf":
                        # Handle union types
                        optimized[key] = [optimize_schema(item, defs) for item in value] if isinstance(value, list) else value
                    elif isinstance(value, (dict, list)):
                        optimized[key] = optimize_schema(value, defs)
                    else:
                        optimized[key] = value
                
                # CRITICAL: Add additionalProperties: false to ALL objects for OpenAI strict mode
                if optimized.get("type") == "object":
                    optimized["additionalProperties"] = False
                
                return optimized
            elif isinstance(obj, list):
                return [optimize_schema(item, defs) for item in obj]
            else:
                return obj
        
        # Optimize the entire schema
        optimized_schema = optimize_schema(schema)
        
        # Ensure root schema has additionalProperties: false if it's an object
        if optimized_schema.get("type") == "object" and "additionalProperties" not in optimized_schema:
            optimized_schema["additionalProperties"] = False
        
        # Fix required array: OpenAI strict mode requires that if 'required' exists,
        # it must include ALL properties. This means all fields in properties must be in required.
        def fix_required_array(obj: Any, defs: Dict[str, Any] = None) -> Any:
            """Fix required arrays to include all properties for OpenAI strict mode."""
            if defs is None:
                defs = defs_lookup
            
            if isinstance(obj, dict):
                fixed = {}
                
                # Handle $ref references
                if "$ref" in obj:
                    ref_path = obj["$ref"]
                    if ref_path.startswith("#/$defs/"):
                        def_name = ref_path.split("/")[-1]
                        if def_name in defs:
                            return fix_required_array(defs[def_name], defs)
                
                # Process all keys
                for key, value in obj.items():
                    if key == "$ref":
                        continue
                    elif isinstance(value, (dict, list)):
                        fixed[key] = fix_required_array(value, defs)
                    else:
                        fixed[key] = value
                
                # Fix required array for object types
                if fixed.get("type") == "object" and "properties" in fixed:
                    properties = fixed.get("properties", {})
                    
                    # OpenAI strict mode requires ALL properties to be in required array
                    # So we add all property keys to required
                    all_property_keys = list(properties.keys())
                    if all_property_keys:
                        fixed["required"] = all_property_keys
                    elif "required" in fixed:
                        # If no properties, keep empty required array
                        fixed["required"] = []
                
                return fixed
            elif isinstance(obj, list):
                return [fix_required_array(item, defs) for item in obj]
            else:
                return obj
        
        # Fix required arrays in the optimized schema
        optimized_schema = fix_required_array(optimized_schema)
        
        # Build the response format dictionary
        response_format_dict = {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_output",
                "strict": True,
                "schema": optimized_schema,
            },
        }
        
        return response_format_dict
    
    def _format_tools(self, tools: Optional[Union[List[Dict], List[Any]]] = None) -> List[Dict[str, Any]]:
        """Format tools for API calls. Convert Tool instances to function call format."""
        if tools is None:
            return []
        if not isinstance(tools, list):
            return tools if isinstance(tools, list) else []
        
        # Check if list contains Tool instances
        # Use duck typing to avoid circular import: check if tool has to_function_call method
        formatted_tools = []
        for tool in tools:
            if hasattr(tool, 'to_function_call') and callable(getattr(tool, 'to_function_call')):
                formatted_tools.append(tool.to_function_call())
            elif isinstance(tool, dict):
                formatted_tools.append(tool)
            else:
                logger.warning(f"Unknown tool type: {type(tool)}, skipping")
        
        return formatted_tools

    def __call__(
        self, 
        messages: List[Message], 
        model_config: ModelConfig,
        tools: Optional[Union[List[Dict], List[Any]]] = None,
        response_format: Optional[Union[BaseModel, Dict, Type[BaseModel]]] = None
    ) -> Dict[str, Any]:
        """
        Format messages, tools, and response_format into a dictionary for API calls.
        
        Args:
            messages: List of messages to format
            model_config: Model configuration
            tools: Optional list of tools (function definitions)
            response_format: Optional response format (BaseModel class/instance or dict)
            
        Returns:
            Dictionary containing:
            - messages: Formatted messages (if not using responses API)
            - input: Input string (if using responses API)
            - tools: Formatted tools (if provided)
            - response_format: Formatted response format (if provided)
        """
        use_responses_api = model_config.use_responses_api
        formatted: List[Dict[str, Any]] = []

        for message in messages:
            role = self._format_role(message)
            content = (
                self._format_content_responses(message)
                if use_responses_api
                else self._format_content_chat(message)
            )
            formatted.append({"role": role, "content": content})

        result: Dict[str, Any] = {}
        
        if use_responses_api:
            # Responses API expects input as a string
            # Extract text from all messages and combine them
            input_parts = []
            for item in formatted:
                content = item.get("content", [])
                if isinstance(content, list):
                    for content_item in content:
                        if isinstance(content_item, dict):
                            text = content_item.get("text", "")
                            if text:
                                input_parts.append(text)
                elif isinstance(content, str):
                    input_parts.append(content)
            result["input"] = "\n".join(input_parts) if input_parts else ""
        else:
            result["messages"] = formatted

        # Process tools if provided
        if tools:
            result["tools"] = self._format_tools(tools)

        # Process response_format if provided
        if response_format:
            if isinstance(response_format, BaseModel) or (isinstance(response_format, type) and issubclass(response_format, BaseModel)):
                # Extract parameter types, descriptions, and default values
                result["response_format"] = self._format_response_format(response_format)
            else:
                # If it's a dict, use it directly
                result["response_format"] = response_format

        return result
        
message_manager = MessageManager()