"""Reporter Tool - A workflow tool for managing and refining markdown reports."""

import os
import json
import re
import uuid
import asyncio
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field, ConfigDict

from src.registry import TOOL
from src.logger import logger
from src.utils import assemble_project_path, dedent
from src.model import model_manager
from src.message import HumanMessage, SystemMessage
from src.tool.types import Tool, ToolResponse, ToolExtra
from src.utils import file_lock


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
    url: Optional[str] = Field(default=None, description="The URL of the reference")
    

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
    report_file_path: Optional[str] = Field(default=None, description="The file path where the report will be saved")

    def __init__(self, model_name: str = None, report_file_path: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if model_name is not None:
            self.model_name = model_name
        if report_file_path is not None:
            self.report_file_path = report_file_path
        
    async def add_item(self, file_path: Optional[str] = None, content: Optional[Union[str, Dict[str, Any]]] = None):
        """Add a new item to the report by extracting ReportItem from content.
        
        Args:
            file_path (Optional[str]): The file path of the content (typically a markdown file). If provided, the file content will be read and appended to the prompt.
            content (Optional[Union[str, Dict[str, Any]]]): Input content as string or dictionary. If string, it will be processed to extract content, summary, and references. If dictionary, it should contain structured data.

        Returns:
            ReportItem: The extracted and added report item
        """
        # Read file content if file_path is provided
        file_content = ""
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
            except Exception as e:
                # If file reading fails, continue without file content
                file_content = f"[Note: Failed to read file {file_path}: {str(e)}]"
        
        # Prepare input text for processing
        if isinstance(content, dict):
            # Convert dict to formatted string
            input_text = json.dumps(content, indent=4, ensure_ascii=False)
        else:
            input_text = str(content) if content else ""
        
        # Combine content and file content
        combined_content = input_text
        if file_content:
            if combined_content:
                combined_content = f"{combined_content}\n\n--- File Content from {file_path} ---\n\n{file_content}"
            else:
                combined_content = f"--- File Content from {file_path} ---\n\n{file_content}"
        
        # Build prompt to extract ReportItem
        prompt = dedent(f"""Extract and structure the following content into a report item with content, summary, and references.
        
        Input Content:
        ```json
        {combined_content}
        ```
        
        Please extract:
        1. **Content**: The main content text (preserve the original content exactly, including all citations in markdown link format [1](url), [2](url), [3](url), etc.)
        2. **Summary**: A concise 2-3 sentence summary of the content
        3. **Reference IDs**: List of integer IDs that reference sources mentioned in the content (e.g., if content has [1](url), [2](url), extract [1, 2])
        4. **References**: List of reference items, each with:
           - id: Integer ID matching the reference IDs found in the content
           - description: Brief description of the reference source (e.g., file path, URL, document name)
           - url: URL for the reference (REQUIRED - extract from content, description, or file_path)
        
        IMPORTANT REQUIREMENTS:
        - **Citation Format**: Citations MUST be in markdown link format: [1](url), [2](url), [3](url), etc. (NOT just [1], [2], [3])
        - **Preserve Citations**: The content field MUST include all citation markers in [number](url) format exactly as they appear in the input
        - **Extract Reference IDs**: Parse all citation numbers from the content (e.g., [1](url), [2](url)
        - **Match References**: Each reference_id in the content must have a corresponding ReferenceItem with matching id
        - **Extract URLs**: Extract URLs from citations (e.g., [1](https://example.com) -> extract "https://example.com") and include them in both:
          - The citation format in content: [1](url)
          - The url field of the corresponding ReferenceItem
        - If the content contains citations like [1](url), [2](url), extract those numbers as reference_ids and create corresponding ReferenceItem entries with matching URLs
        - If citations are in [1] format without URLs, convert them to [1](url) format using the URL from the corresponding reference
        - If no citations are present, you may infer references from the content or use empty lists
        - If file_path is provided, include it in the references with an appropriate description and URL (convert file path to file:// URL format)
        
        Please Only Return the ReportItem object, no other text or explanation.
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
        
        if not response.success or not getattr(response, 'extra', None):
            raise ValueError(f"Failed to extract report item: {response.message}")
        
        # Safely access parsed_model using getattr
        report_item = response.extra.parsed_model
        
        # Add to items list
        self.items.append(report_item)
        
        return report_item
    
    async def complete(self):
        """Complete the report by optimizing the content and references.
        
        This method:
        1. Collects all items from the report
        2. Merges and deduplicates all references
        3. Renumbers citations in content and references
        4. Uses LLM to generate a complete markdown report
        5. Writes the report to the file path specified during initialization
        
        Raises:
            ValueError: If report_file_path is not set or report has no items
        """
        if not self.items:
            raise ValueError("Cannot complete report: no items found")
        
        if not self.report_file_path:
            raise ValueError("Cannot complete report: report_file_path is not set")
        
        # Step 1: Collect all unique references from all items
        # Deduplicate by both description and URL to handle similar references
        all_references_dict: Dict[str, ReferenceItem] = {}  # normalized_key -> ReferenceItem
        reference_key_to_id: Dict[str, int] = {}  # normalized_key -> first_seen_id
        
        def normalize_reference_key(ref: ReferenceItem) -> str:
            """Create a normalized key for deduplication based on description and URL."""
            # Normalize description: strip whitespace, lowercase for comparison
            desc = ref.description.strip().lower() if ref.description else ""
            
            # Normalize URL: extract and normalize URL
            url = ref.url.strip().lower() if ref.url else ""
            
            # If URL exists, use URL as primary key (more reliable for deduplication)
            if url:
                # Normalize URL: remove trailing slashes, convert to lowercase
                url_normalized = url.rstrip('/')
                return f"url:{url_normalized}"
            
            # If no URL but description looks like a URL, use it
            if desc.startswith(('http://', 'https://', 'file://')):
                desc_normalized = desc.rstrip('/')
                return f"url:{desc_normalized}"
            
            # Otherwise, use normalized description
            return f"desc:{desc}"
        
        for item in self.items:
            for ref in item.references:
                normalized_key = normalize_reference_key(ref)
                
                # If we've seen this reference before, merge information
                if normalized_key in all_references_dict:
                    existing_ref = all_references_dict[normalized_key]
                    # Prefer non-empty values: use URL if available, otherwise keep existing
                    if ref.url and not existing_ref.url:
                        existing_ref.url = ref.url
                    # Prefer more descriptive description
                    if ref.description and len(ref.description) > len(existing_ref.description):
                        existing_ref.description = ref.description
                else:
                    # First time seeing this reference
                    all_references_dict[normalized_key] = ref
                    reference_key_to_id[normalized_key] = ref.id
        
        # Step 2: Create new reference mapping (old_id -> new_id)
        unique_references = list(all_references_dict.values())
        reference_mapping: Dict[int, int] = {}  # old_id -> new_id
        
        # Create mapping based on normalized key
        for new_id, (normalized_key, ref) in enumerate(all_references_dict.items(), start=1):
            # Find all old IDs that map to this reference (by normalized key)
            for item in self.items:
                for old_ref in item.references:
                    old_normalized_key = normalize_reference_key(old_ref)
                    if old_normalized_key == normalized_key:
                        reference_mapping[old_ref.id] = new_id
        
        # Step 3: Create URL mapping for references (new_id -> url)
        reference_urls: Dict[int, str] = {}  # new_id -> url
        for new_id, ref in enumerate(unique_references, start=1):
            description = ref.description
            # Generate URL from description
            # If description is already a URL (http/https), use it directly
            if description.startswith(('http://', 'https://')):
                reference_urls[new_id] = description
            # If description is a file path, convert to file:// URL
            elif os.path.exists(description) or '/' in description or '\\' in description:
                # Convert to absolute path if relative
                abs_path = os.path.abspath(description) if not os.path.isabs(description) else description
                reference_urls[new_id] = f"file://{abs_path}"
            # Otherwise, try to extract URL from description or use description as-is
            else:
                # Try to find URL pattern in description
                url_match = re.search(r'(https?://[^\s]+)', description)
                if url_match:
                    reference_urls[new_id] = url_match.group(1)
                else:
                    # Use description as URL (fallback)
                    reference_urls[new_id] = description
        
        # Step 4: Update all items' content citations and reference_ids
        updated_contents = []
        for item in self.items:
            content = item.content.content
            reference_ids = item.content.reference_ids
            
            # Update citations in content: [old_id] -> [new_id](url)
            def replace_citation(match):
                old_id_str = match.group(1)
                try:
                    old_id = int(old_id_str)
                    new_id = reference_mapping.get(old_id)
                    if new_id is not None:
                        url = reference_urls.get(new_id, f"#ref{new_id}")
                        return f"[{new_id}]({url})"
                    return match.group(0)  # Keep original if not found
                except ValueError:
                    return match.group(0)  # Keep original if not a number
            
            # Replace citations in content using regex
            # Handle both [1] and [1](url) formats
            updated_content = re.sub(r'\[(\d+)\]?(?:\([^)]+\))?', replace_citation, content)
            
            # Update reference_ids
            updated_reference_ids = [reference_mapping.get(rid, rid) for rid in reference_ids]
            # Remove duplicates and sort
            updated_reference_ids = sorted(list(set(updated_reference_ids)))
            
            updated_contents.append({
                "content": updated_content,
                "summary": item.content.summary,
                "reference_ids": updated_reference_ids
            })
        
        # Step 5: Create renumbered references list with URLs
        renumbered_references = []
        for new_id, ref in enumerate(unique_references, start=1):
            # Use ref.url if available, otherwise use generated URL from reference_urls
            url = ref.url if ref.url else reference_urls.get(new_id, ref.description)
            renumbered_references.append({
                "id": new_id,
                "description": ref.description,
                "url": url
            })
        
        # Step 6: Build prompt for LLM to generate final report
        items_text = "\n\n".join([
            f"## Item {i+1}\n\n**Summary:** {item['summary']}\n\n**Content:**\n{item['content']}\n\n**Reference IDs:** {item['reference_ids']}"
            for i, item in enumerate(updated_contents)
        ])
        
        # Generate references text with each reference on its own line
        references_text = "\n".join([
            f"[{ref['id']}]({ref['url']}) {ref['description']}"
            for ref in renumbered_references
        ])
        # Ensure proper spacing - add newline before and after, and double newline between items
        if references_text:
            references_text = "\n" + references_text.replace("\n", "\n\n") + "\n"
        # Ensure each reference is on its own line with proper spacing
        if references_text:
            references_text = "\n" + references_text.replace("\n", "\n\n") + "\n"
        
        prompt = dedent(f"""Generate a complete, well-structured markdown report based on the following report items and references.
        
        Report Title: {self.title}
        
        Report Items:
        {items_text}
        
        References:
        {references_text}
        
        Please generate a comprehensive markdown report that:
        1. **Starts with the title** as a main heading (# {self.title})
        2. **Organizes content logically** - Group related items into sections with appropriate headings
        3. **Preserves all citations** - Keep all citation markers [1](url), [2](url), [3](url), etc. exactly as they appear in the content (with URLs)
        4. **Integrates summaries** - Use item summaries to create smooth transitions and context
        5. **Maintains coherence** - Ensure the report flows logically from introduction to conclusion
        6. **Includes References section** - Add a "## References" section at the end listing all references in numerical order with URLs, each on a separate line with proper spacing:
           ```
           ## References
           
           [1](url1) Reference description 1
           
           [2](url2) Reference description 2
           
           ...
           ```
        
        IMPORTANT REQUIREMENTS:
        - **Preserve All Citations**: Keep all citation markers [1](url), [2](url), [3](url) exactly as they appear in the content (with URLs)
        - **Citation Format**: All citations MUST be in markdown link format: [number](url)
        - **Preserve All Facts**: Do not modify facts, numbers, data, or specific details from the content
        - **Use All Content**: Include all content from all items, organized logically
        - **Complete References**: Include all references in the References section, numbered sequentially [1](url), [2](url), [3](url), etc.
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
        
        # Step 7: Call LLM to generate the report
        response = await model_manager(
            model=self.model_name,
            messages=messages
        )
        
        if not response.success:
            raise ValueError(f"Failed to generate report: {response.message}")
        
        report_content = response.message.strip()
        
        # Step 8: Ensure References section exists and is properly formatted
        if "## References" not in report_content and "References" not in report_content:
            # Add References section if missing
            report_content += f"\n\n## References\n\n{references_text}\n"
        else:
            # Verify References section has correct format with URLs
            # Replace any existing references section with properly formatted one
            report_content = re.sub(
                r'## References.*?(?=\n##|\Z)',
                f'## References\n\n{references_text}\n',
                report_content,
                flags=re.DOTALL
            )
        
        # Step 9: Ensure all citations have URLs: [number] -> [number](url)
        def add_url_to_citation(match):
            citation_num = match.group(1)
            # Check if this citation already has a URL
            if match.group(0).count('(') == 0:  # No URL found
                # Find the URL for this citation number
                citation_id = int(citation_num)
                url = reference_urls.get(citation_id, f"#ref{citation_id}")
                return f"[{citation_num}]({url})"
            return match.group(0)  # Already has URL, keep as is
        
        # Replace [number] with [number](url) if not already in that format
        report_content = re.sub(r'\[(\d+)\](?!\()', add_url_to_citation, report_content)
        
        # Step 10: Write report to file with file lock for concurrent safety
        os.makedirs(os.path.dirname(self.report_file_path), exist_ok=True)
        async with file_lock(self.report_file_path):
            with open(self.report_file_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
        
        return report_content


_REPORT_DESCRIPTION = """Report tool for managing and refining markdown reports.

🎯 BEST FOR: Creating, editing, and refining analysis reports.

📋 Actions:
- add: Add new content to the report
  - args: 
    - call_id (str) - REQUIRED: Unique identifier for the report. Use the same call_id across multiple calls to maintain state.
    - file_path (Optional[str]) - Path to a markdown file to add (file content will be read and added)
    - content (Optional[Union[str, Dict[str, Any]]]) - The content to add as string or dictionary
    - At least one of content or file_path must be provided
  - Automatically generates summary and updates content list
  - Appends content to report.md

- complete: Complete and optimize the entire report
  - args:
    - call_id (str) - REQUIRED: Unique identifier for the report. Must match the call_id used in previous 'add' calls.
  - Reads all summaries and optimizes content for coherence and logic
  - Updates report.md with optimized content

💡 Workflow:
1. Use `add` multiple times with the SAME `call_id` to incrementally add content (from strings or dictionaries or files)
2. Use `complete` with the SAME `call_id` to optimize the entire report with LLM

⚠️ IMPORTANT: You MUST use the same `call_id` for all calls to the same report. Different `call_id` values create separate reports.
"""


@TOOL.register_module(force=True)
class ReporterTool(Tool):
    """A tool for managing and refining markdown reports."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = "reporter"
    description: str = _REPORT_DESCRIPTION
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the tool")
    require_grad: bool = Field(default=False, description="Whether the tool requires gradients")
    
    model_name: str = Field(
        default="openrouter/gemini-3-flash-preview",
        description="The model to use for code generation."
    )

    # Configuration parameters
    base_dir: str = Field(
        default="workdir/reporter",
        description="The base directory for saving reports."
    )
    
    def __init__(
        self, 
        base_dir: Optional[str] = None, 
        model_name: Optional[str] = None,
        require_grad: bool = False,
        **kwargs
    ):
        """Initialize the reporter tool."""
        super().__init__(require_grad=require_grad, **kwargs)
        
        if model_name is not None:
            self.model_name = model_name
        
        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
        else:
            self.base_dir = assemble_project_path(self.base_dir)
            
        if self.base_dir is not None:
            os.makedirs(self.base_dir, exist_ok=True)
        
        # Per-call_id cache and locks for concurrent safety (similar to memory system)
        # Key: call_id (str), Value: Report instance
        self._report_cache: Dict[str, Report] = {}
        # Key: call_id (str), Value: asyncio.Lock for that report
        self._report_locks: Dict[str, asyncio.Lock] = {}
        # Lock for managing the cache dictionaries themselves
        self._cache_lock = asyncio.Lock()
    
    def _get_report_file_path(self, call_id: str) -> Optional[str]:
        """Generate a fixed report file path based on call_id.
        
        Args:
            call_id: The unique call identifier
            
        Returns:
            The report file path, or None if base_dir is not set
        """
        if not self.base_dir:
            return None
        # Create a safe filename from call_id
        safe_id = re.sub(r'[^\w\s-]', '', call_id).strip().replace(' ', '_')
        if not safe_id:
            safe_id = "report"
        md_filename = f"{safe_id}.md"
        return os.path.join(self.base_dir, md_filename)
    
    async def _get_or_create_report(self, call_id: str) -> tuple[Report, asyncio.Lock]:
        """Get or create a Report instance for the given call_id with proper locking.
        
        Args:
            call_id: The unique identifier for the call
            
        Returns:
            tuple[Report, asyncio.Lock]: The report instance and its lock
        """
        async with self._cache_lock:
            # Get or create lock for this call_id
            if call_id not in self._report_locks:
                self._report_locks[call_id] = asyncio.Lock()
            
            # Get or create report for this call_id
            if call_id not in self._report_cache:
                # Generate report_file_path based on call_id
                report_file_path = self._get_report_file_path(call_id)
                # Use call_id as the title
                self._report_cache[call_id] = Report(
                    title=call_id,
                    model_name=self.model_name,
                    report_file_path=report_file_path
                )
                logger.info(f"| 📝 Created new report cache for id: {call_id} (file_path: {report_file_path})")
            else:
                logger.info(f"| 📂 Using existing report cache for id: {call_id} (items: {len(self._report_cache[call_id].items)})")
            
            return self._report_cache[call_id], self._report_locks[call_id]
    
    async def _cleanup_report(self, call_id: str):
        """Remove report from cache after completion."""
        async with self._cache_lock:
            if call_id in self._report_cache:
                del self._report_cache[call_id]
                logger.info(f"| 🧹 Removed report from cache: {call_id}")
            if call_id in self._report_locks:
                del self._report_locks[call_id]

    async def __call__(
        self,
        action: str,
        call_id: str,
        file_path: Optional[str] = None,
        content: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> ToolResponse:
        """Execute report action.

        Args:
            action (str): The action to perform. action must be one of: add, complete.
            call_id (str): REQUIRED: Unique identifier for the report. Use the same call_id across multiple calls to maintain state.
            file_path (Optional[str]): Path to a file to add. If it's a .md file, the content will be read and added. If it's not a .md file, it will be added as a reference/attachment.
            content (Optional[Union[str, Dict[str, Any]]]): Content to add as string or dictionary.
        """
        try:
            if not call_id:
                return ToolResponse(
                    success=False,
                    message="call_id is required. Please provide a unique identifier for the report."
                )
            
            if not self.base_dir:
                return ToolResponse(
                    success=False,
                    message="Cannot create report: base_dir is not set."
                )
            
            # Get or create report instance and its lock based on call_id
            # report_file_path is set during Report initialization
            report, report_lock = await self._get_or_create_report(call_id)
            
            # Use the lock to ensure thread-safe access to this specific report
            async with report_lock:
                logger.info(f"| 📝 ReporterTool action: {action} (call_id: {call_id}, report_file_path: {report.report_file_path}, items: {len(report.items)})")

                if action == "add":
                    if not content and not file_path:
                        return ToolResponse(
                            success=False,
                            message="At least one of 'content' or 'file_path' is required for add action."
                        )
                    return await self._add_content(report=report, file_path=file_path, content=content)

                elif action == "complete":
                    result = await self._complete_report(report=report)
                    # Clean up cache after completion
                    await self._cleanup_report(call_id)
                    return result

                else:
                    return ToolResponse(
                        success=False,
                        message=f"Unknown action: {action}. Valid actions: add, complete"
                    )

        except Exception as e:
            logger.error(f"| ❌ Error in ReporterTool: {e}")
            import traceback
            return ToolResponse(
                success=False,
                message=f"Error in report action '{action}': {str(e)}\n{traceback.format_exc()}"
            )

    async def _add_content(self, report: Report, file_path: Optional[str] = None, content: Optional[Union[str, Dict[str, Any]]] = None) -> ToolResponse:
        """Add new content to the report using Report.add_item().
        
        Args:
            report: The report instance to add content to.
            file_path (Optional[str]): Path to a file to add. 
                - If it's a .md file, the content will be read and added.
                - If it's not a .md file, it will be added as a reference/attachment.
            content (Optional[Union[str, Dict[str, Any]]]): Content to add as string or dictionary.
        """
        try:
            # Resolve file_path to absolute path if provided
            resolved_file_path = None
            is_markdown_file = False
            
            if file_path:
                if os.path.isabs(file_path):
                    resolved_file_path = file_path
                else:
                    # Try relative to base_dir first, then current directory
                    if self.base_dir:
                        potential_path = os.path.join(self.base_dir, file_path)
                        if os.path.exists(potential_path):
                            resolved_file_path = os.path.abspath(potential_path)
                        else:
                            resolved_file_path = os.path.abspath(file_path)
                    else:
                        resolved_file_path = os.path.abspath(file_path)
                
                # Check if file is a markdown file
                if resolved_file_path and os.path.exists(resolved_file_path):
                    file_ext = os.path.splitext(resolved_file_path)[1].lower()
                    is_markdown_file = file_ext in ['.md', '.markdown']
            
            # If it's a markdown file, read content and pass file_path to add_item
            # If it's not a markdown file, just pass it as file_path (will be added as reference)
            if is_markdown_file:
                # Markdown file: read content and pass file_path to add_item
                report_item = await report.add_item(file_path=resolved_file_path, content=content)
            else:
                # Non-markdown file: pass file_path but don't read content (will be added as reference)
                # Create a note about the attached file
                file_note = f"Attached file: {resolved_file_path}" if resolved_file_path else ""
                combined_content = content if content else ""
                if file_note:
                    if combined_content:
                        combined_content = f"{combined_content}\n\n{file_note}"
                    else:
                        combined_content = file_note
                
                report_item = await report.add_item(file_path=resolved_file_path, content=combined_content)
            
            item_id = len(report.items)
            logger.info(f"| ✅ Content added: ID={item_id}, Summary={report_item.content.summary[:100]}...")
            
            # Build success message
            message_parts = [f"📝 Content added successfully!\n\nID: {item_id}\nSummary: {report_item.content.summary}"]
            if resolved_file_path:
                message_parts.append(f"\nFile: {resolved_file_path}")
            
            return ToolResponse(
                success=True,
                message="\n".join(message_parts),
                extra=ToolExtra(
                    file_path=report.report_file_path,
                    data={
                        "id": item_id,
                        "summary": report_item.content.summary,
                        "reference_ids": report_item.content.reference_ids,
                        "references": [{"id": ref.id, "description": ref.description, "url": ref.url} for ref in report_item.references],
                        "total_items": len(report.items),
                        "source_file_path": resolved_file_path if resolved_file_path else None
                    }
                )
            )

        except Exception as e:
            logger.error(f"| ❌ Error adding content: {e}")
            import traceback
            return ToolResponse(
                success=False,
                message=f"Error adding content: {str(e)}\n{traceback.format_exc()}"
            )

    async def _complete_report(self, report: Report) -> ToolResponse:
        """Complete and optimize the entire report using Report.complete()."""
        try:
            if not report or not report.items:
                return ToolResponse(
                    success=False,
                    message="Report is empty. Add content first using the 'add' action."
                )
            
            if not report.report_file_path:
                return ToolResponse(
                    success=False,
                    message="Report file path is not set. Cannot complete report."
                )
            
            logger.info(f"| 📊 Completing report with {len(report.items)} items...")
            
            # Use Report.complete() to generate final report with renumbered citations and references
            final_report_content = await report.complete()
            
            logger.info(f"| ✅ Report completion successful ({len(final_report_content)} chars)")
            
            return ToolResponse(
                success=True,
                message=f"📝 Report completed successfully!\n\nPath: {report.report_file_path}\n\nThe entire report has been generated with properly numbered citations and references.",
                extra=ToolExtra(
                    file_path=report.report_file_path,
                    data={
                        "path": report.report_file_path,
                        "items_count": len(report.items),
                        "report_length": len(final_report_content),
                        "title": report.title
                    }
                )
            )

        except Exception as e:
            logger.error(f"| ❌ Error completing report: {e}")
            import traceback
            return ToolResponse(
                success=False,
                message=f"Error completing report: {str(e)}\n{traceback.format_exc()}"
            )

