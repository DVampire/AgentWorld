from __future__ import annotations

import inspect
import json
import os
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Type, get_type_hints, Union

from pydantic import BaseModel, ConfigDict, Field

from src.utils import (
    PYTHON_TYPE_FIELD,
    default_parameters_schema,
    parse_docstring_descriptions,
    annotation_to_types,
    build_args_schema,
    build_function_calling,
    build_text_representation,
    dedent,
)
from src.model import model_manager
from src.message import HumanMessage, SystemMessage

class ContentItem(BaseModel):
    """Content item"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    content: str = Field(description="The content of the item")
    summary: str = Field(description="The summary of the item")
    reference_ids: List[int] = Field(description="The reference IDs of the item")
    
class ReferenceItem(BaseModel):
    """Reference item"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    id: int = Field(description="The ID of the reference")
    description: str = Field(description="The brief description of the reference")
    
class ReportItem(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    content: ContentItem = Field(description="The content of the item")
    references: List[ReferenceItem] = Field(description="The references of the item")
    
class Report(BaseModel):
    """Report"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    title: str = Field(description="The title of the report")
    items: List[ReportItem] = Field(default=[], description="The items of the report")
    model_name: str = Field(default="openrouter/gemini-3-flash-preview", description="The model to use for extraction")

    def __init__(self, model_name: str = None, **kwargs):
        super().__init__(**kwargs)
        if model_name is not None:
            self.model_name = model_name
        
    async def add_item(self, content: Union[str, Dict[str, Any]]):
        """Add a new item to the report by extracting ReportItem from content.
        
        Args:
            content: Input content as string or dictionary. If string, it will be processed to extract
                     content, summary, and references. If dictionary, it should contain structured data.
        
        Returns:
            ReportItem: The extracted and added report item
        """
        # Prepare input text for processing
        if isinstance(content, dict):
            # Convert dict to formatted string
            input_text = json.dumps(content, indent=4, ensure_ascii=False)
        else:
            input_text = str(content)
        
        # Build prompt to extract ReportItem
        prompt = dedent(f"""Extract and structure the following content into a report item with content, summary, and references.
        
        Input Content:
        ```json
        {input_text}
        ```
        
        Please extract:
        1. **Content**: The main content text (preserve the original content exactly, including all citations like [1], [2], [3], etc.)
        2. **Summary**: A concise 2-3 sentence summary of the content
        3. **Reference IDs**: List of integer IDs that reference sources mentioned in the content (e.g., if content has [1], [2], extract [1, 2])
        4. **References**: List of reference items, each with:
           - id: Integer ID matching the reference IDs found in the content
           - description: Brief description of the reference source (e.g., file path, URL, document name)
        
        IMPORTANT REQUIREMENTS:
        - **Preserve Citations**: The content field MUST include all citation markers like [1], [2], [3] exactly as they appear in the input
        - **Extract Reference IDs**: Parse all citation numbers from the content (e.g., [1], [2] -> extract [1, 2] as reference_ids)
        - **Match References**: Each reference_id in the content must have a corresponding ReferenceItem with matching id
        - If the content contains citations like [1], [2], extract those numbers as reference_ids and create corresponding ReferenceItem entries
        - If no citations are present, you may infer references from the content or use empty lists
        
        Return a structured ReportItem with:
        - content: ContentItem containing the main content (with citations preserved), summary, and reference_ids
        - references: List of ReferenceItem objects with id and description matching the citations in content
        """)
        
        messages = [
            SystemMessage(content="You are an expert at extracting structured information from content. Extract content, summaries, and references accurately."),
            HumanMessage(content=prompt)
        ]
        
        # Call model_manager with ReportItem as response_format
        response = await model_manager(
            model=self.model_name,
            messages=messages,
            response_format=ReportItem
        )
        
        # Extract parsed model from response
        if not response.success:
            raise ValueError(f"Failed to extract report item: {response.message}")
        
        if "parsed_model" not in (response.extra or {}):
            raise ValueError("Response does not contain parsed_model")
        
        report_item = response.extra["parsed_model"]
        
        # Add to items list
        self.items.append(report_item)
        
        return report_item
    
    async def complete(self, report_path: str):
        """Complete the report by optimizing the content and references.
        
        This method:
        1. Collects all items from the report
        2. Merges and deduplicates all references
        3. Renumbers citations in content and references
        4. Uses LLM to generate a complete markdown report
        5. Writes the report to the specified path
        
        Args:
            report_path: Path to write the final markdown report file
        """
        if not self.items:
            raise ValueError("Cannot complete report: no items found")
        
        # Step 1: Collect all unique references from all items
        all_references_dict: Dict[str, ReferenceItem] = {}  # description -> ReferenceItem
        for item in self.items:
            for ref in item.references:
                # Use description as key to deduplicate
                if ref.description not in all_references_dict:
                    all_references_dict[ref.description] = ref
        
        # Step 2: Create new reference mapping (old_id -> new_id)
        unique_references = list(all_references_dict.values())
        reference_mapping: Dict[int, int] = {}  # old_id -> new_id
        
        # Create mapping based on description order
        for new_id, ref in enumerate(unique_references, start=1):
            # Find all old IDs that map to this reference
            for item in self.items:
                for old_ref in item.references:
                    if old_ref.description == ref.description:
                        reference_mapping[old_ref.id] = new_id
        
        # Step 3: Update all items' content citations and reference_ids
        updated_contents = []
        for item in self.items:
            content = item.content.content
            reference_ids = item.content.reference_ids
            
            # Update citations in content: [old_id] -> [new_id]
            def replace_citation(match):
                old_id_str = match.group(1)
                try:
                    old_id = int(old_id_str)
                    new_id = reference_mapping.get(old_id)
                    if new_id is not None:
                        return f"[{new_id}]"
                    return match.group(0)  # Keep original if not found
                except ValueError:
                    return match.group(0)  # Keep original if not a number
            
            # Replace citations in content using regex
            updated_content = re.sub(r'\[(\d+)\]', replace_citation, content)
            
            # Update reference_ids
            updated_reference_ids = [reference_mapping.get(rid, rid) for rid in reference_ids]
            # Remove duplicates and sort
            updated_reference_ids = sorted(list(set(updated_reference_ids)))
            
            updated_contents.append({
                "content": updated_content,
                "summary": item.content.summary,
                "reference_ids": updated_reference_ids
            })
        
        # Step 4: Create renumbered references list
        renumbered_references = []
        for new_id, ref in enumerate(unique_references, start=1):
            renumbered_references.append({
                "id": new_id,
                "description": ref.description
            })
        
        # Step 5: Build prompt for LLM to generate final report
        items_text = "\n\n".join([
            f"## Item {i+1}\n\n**Summary:** {item['summary']}\n\n**Content:**\n{item['content']}\n\n**Reference IDs:** {item['reference_ids']}"
            for i, item in enumerate(updated_contents)
        ])
        
        references_text = "\n".join([
            f"[{ref['id']}] {ref['description']}"
            for ref in renumbered_references
        ])
        
        prompt = dedent(f"""Generate a complete, well-structured markdown report based on the following report items and references.
        
        Report Title: {self.title}
        
        Report Items:
        {items_text}
        
        References:
        {references_text}
        
        Please generate a comprehensive markdown report that:
        1. **Starts with the title** as a main heading (# {self.title})
        2. **Organizes content logically** - Group related items into sections with appropriate headings
        3. **Preserves all citations** - Keep all citation markers [1], [2], [3], etc. exactly as they appear in the content
        4. **Integrates summaries** - Use item summaries to create smooth transitions and context
        5. **Maintains coherence** - Ensure the report flows logically from introduction to conclusion
        6. **Includes References section** - Add a "## References" section at the end listing all references in numerical order:
           ```
           ## References
           [1] Reference description 1
           [2] Reference description 2
           ...
           ```
        
        IMPORTANT REQUIREMENTS:
        - **Preserve All Citations**: Keep all citation markers [1], [2], [3] exactly as they appear in the content
        - **Preserve All Facts**: Do not modify facts, numbers, data, or specific details from the content
        - **Use All Content**: Include all content from all items, organized logically
        - **Complete References**: Include all references in the References section, numbered sequentially [1], [2], [3], etc.
        - **Markdown Format**: Use proper markdown formatting (headings, lists, paragraphs, etc.)
        - **Professional Style**: Write in a professional, academic report style
        
        ⚠️ CRITICAL FILE PATH REQUIREMENTS:
        - **MUST use absolute paths** for all file references in markdown content (images, links, file paths, etc.)
        - When referencing images or files in the report content, use absolute paths like:
          - ✅ Correct: `![Chart](/path/to/workdir/esg_agent/tool/plotter/chart.png)`
          - ✅ Correct: `[Link](/path/to/workdir/esg_agent/tool/data/file.pdf)`
          - ❌ Wrong: `![Chart](chart.png)` or `![Chart](./chart.png)` or `![Chart](../chart.png)`
        - Absolute paths ensure proper rendering in markdown viewers and editors
        - If any file paths appear in the content or references, they MUST be absolute paths
        
        Return ONLY the complete markdown report content, no explanations or additional text.
        """)
        
        messages = [
            SystemMessage(content="You are an expert report writer specializing in creating comprehensive, well-structured reports with proper citations and references."),
            HumanMessage(content=prompt)
        ]
        
        # Step 6: Call LLM to generate the report
        response = await model_manager(
            model=self.model_name,
            messages=messages
        )
        
        if not response.success:
            raise ValueError(f"Failed to generate report: {response.message}")
        
        report_content = response.message.strip()
        
        # Step 7: Ensure References section exists and is properly formatted
        if "## References" not in report_content and "References" not in report_content:
            # Add References section if missing
            report_content += f"\n\n## References\n\n{references_text}\n"
        else:
            # Verify References section has correct format
            # Replace any existing references section with properly formatted one
            report_content = re.sub(
                r'## References.*?(?=\n##|\Z)',
                f'## References\n\n{references_text}\n',
                report_content,
                flags=re.DOTALL
            )
        
        # Step 8: Write report to file
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return report_content

class ToolResponse(BaseModel):
    """Response for a tool call."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    success: bool = Field(description="Whether the tool call was successful")
    message: str = Field(description="The message from the tool call")
    extra: Optional[Dict[str, Any]] = Field(default=None, description="The extra data from the tool call")
    
    def __str__(self) -> str:
        return f"ToolResponse(success={self.success}, message={self.message}, extra={self.extra})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Dump the model to a dictionary, recursively serializing nested Pydantic models."""
        from pydantic import BaseModel
        
        def serialize_value(value: Any) -> Any:
            """Recursively serialize Pydantic models and other nested structures."""
            if isinstance(value, BaseModel):
                return value.model_dump(**kwargs)
            elif isinstance(value, list):
                return [serialize_value(item) for item in value]
            elif isinstance(value, dict):
                return {k: serialize_value(v) for k, v in value.items()}
            else:
                return value
        
        result = {
            "success": self.success,
            "message": self.message,
            "extra": serialize_value(self.extra) if self.extra is not None else None
        }
        return result
    
    def model_dump_json(self) -> str:
        return json.dumps(self.model_dump())

class Tool(BaseModel):
    """Base class for all tools that can be exposed through function calling."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(description="The name of the tool")
    description: str = Field(description="The description of the tool")
    enabled: bool = Field(default=True, description="Whether the tool is enabled")

    @staticmethod
    def default_parameters_schema() -> Dict[str, Any]:
        return default_parameters_schema()
        
    @property
    def name(self) -> str:
        return self._name
    
    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def description(self) -> str:
        return self._description

    @description.setter
    def description(self, value: str) -> None:
        self._description = value
        
    @property
    def is_enabled(self) -> bool:
        return bool(self.enabled)

    @is_enabled.setter
    def is_enabled(self, value: bool) -> None:
        self.enabled = bool(value)

    @property
    def parameter_schema(self) -> Dict[str, Any]:
        schema = self._build_parameter_schema()
        return deepcopy(schema)

    async def __call__(self, input: Dict[str, Any], **kwargs) -> ToolResponse:
        """
        Execute the tool asynchronously.
        
        Args:
            input (Dict[str, Any]): The input to the tool.
            
        Returns:
            ToolResponse: The response from the tool call.
        """
        raise NotImplementedError("Tool subclasses must implement __call__")
    
    @property
    def function_calling(self) -> Dict[str, Any]:
        """Return the OpenAI-compatible function-calling representation."""
        schema = self.parameter_schema
        return build_function_calling(self.name, self.description, schema)
        
    @property
    def args_schema(self) -> Type[BaseModel]:
        """Return a BaseModel type for the tool's input parameters.
        
        The model name will be `{tool_name}Input` (e.g., `bashInput`, `python_interpreterInput`).
        
        Returns:
            Type[BaseModel]: A Pydantic BaseModel class for the tool's input parameters
        """
        schema = self.parameter_schema
        return build_args_schema(self.name, schema)
    
    @property
    def text(self) -> str:
        """
        Return the text representation of the tool.
        
        Example:
        ```
        Tool: python_interpreter
        Description: A tool that can execute Python code.
        Parameters:
            - code (str): Python code to execute. (default: None)
        ```
        """
        schema = self.parameter_schema
        return build_text_representation(self.name, self.description, schema, entity_type="Tool")

    def _build_parameter_schema(self) -> Dict[str, Any]:
        """Build parameter schema from function signature and docstring."""
        try:
            signature = inspect.signature(self.__class__.__call__)
        except (TypeError, ValueError):
            return self.default_parameters_schema()

        # Get type hints
        try:
            hints = get_type_hints(self.__class__.__call__)
        except Exception:
            hints = {}

        # Get docstring for descriptions
        docstring = inspect.getdoc(self.__class__.__call__) or ""
        doc_descriptions = parse_docstring_descriptions(docstring)

        properties = {}
        required = []
        
        for name, param in signature.parameters.items():
            if name == "self":
                continue
            
            # Skip generic "input" parameter
            if name == "input" and len(signature.parameters) == 2:  # self + input
                continue
            
            # Skip VAR_KEYWORD (**kwargs) and VAR_POSITIONAL (*args) parameters
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            
            # Get type annotation
            annotation = hints.get(name, param.annotation)
            json_type, python_type = annotation_to_types(annotation)
            
            # Determine if required
            is_required = param.default is inspect._empty
            
            # Build schema
            schema: Dict[str, Any] = {
                "type": json_type,
                "description": doc_descriptions.get(name, ""),
            }
            schema[PYTHON_TYPE_FIELD] = python_type
            
            if not is_required:
                schema["default"] = param.default
            
            properties[name] = schema
            if is_required:
                required.append(name)

        if not properties:
            return self.default_parameters_schema()

        result: Dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        }
        if required:
            result["required"] = required
        return result

class ToolConfig(BaseModel):
    """Tool configuration"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(description="The name of the tool")
    description: str = Field(description="The description of the tool")
    enabled: bool = Field(default=True, description="Whether the tool is enabled")
    version: str = Field(default="1.0.0", description="Version of the tool")
    
    cls: Optional[Type[Tool]] = Field(default=None, description="The class of the tool")
    config: Optional[Dict[str, Any]] = Field(default={}, description="The initialization configuration of the tool")
    instance: Optional[Tool] = Field(default=None, description="The instance of the tool")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="The metadata of the tool")
    code: Optional[str] = Field(default=None, description="Source code for dynamically generated tool classes (used when cls cannot be imported from a module)")
    
    # Default representations
    function_calling: Optional[Dict[str, Any]] = Field(default=None, description="Default function calling representation")
    text: Optional[str] = Field(default=None, description="Default text representation")
    args_schema: Optional[Type[BaseModel]] = Field(default=None, description="Default args schema (BaseModel type)")

__all__ = [
    "Tool",
    "ToolResponse",
    "ToolConfig",
]

