"""Mdify tool for converting various file formats to markdown text."""

import asyncio
import os
from typing import Optional
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from typing import Type, Dict, Any

from src.tools.protocol.tool import ToolResponse
from src.tools.protocol import tcp
from src.tools.default_tools.markdown.mdconvert import MarkitdownConverter
from src.logger import logger

_MDIFY_TOOL_DESCRIPTION = """Convert various file formats to markdown text using markitdown.
This tool can convert files to markdown format for easy text processing and analysis.
The input should be a file path (absolute path recommended) to the file you want to convert.

Supported file formats:
- Documents: PDF, DOCX, PPTX, XLSX, XLS, CSV, TXT, HTML, EPUB
- Images: PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP (with OCR text extraction)
- Audio: MP3, WAV, MP4, M4A (with speech-to-text transcription)
- Archives: ZIP (extracts and converts contents)
- Data: IPYNB (Jupyter notebooks), RSS feeds
- Email: MSG (Outlook messages)
- Web: Wikipedia pages, YouTube videos (metadata and transcripts)
- Plain text files

The tool will extract text content, tables, metadata, and other structured information from these files and convert them into readable markdown format.
"""

class MdifyToolArgs(BaseModel):
    file_path: str = Field(description="The absolute path to the file to convert to markdown")
    output_format: Optional[str] = Field(
        default="markdown", 
        description="Output format (default: markdown)"
    )

@tcp.tool()
class MdifyTool(BaseTool):
    """A tool for converting various file formats to markdown text asynchronously."""
    name: str = "mdify"
    description: str = _MDIFY_TOOL_DESCRIPTION
    args_schema: Type[BaseModel] = MdifyToolArgs
    metadata: Dict[str, Any] = {"type": "Markdown Conversion"}
    
    timeout: int = Field(description="Timeout in seconds for file conversion", default=60)
    converter: MarkitdownConverter = Field(description="The converter to use for file conversion", default=None)
    
    def __init__(self,  **kwargs):
        """A tool for converting various file formats to markdown text asynchronously"""
        super().__init__(**kwargs)
        self.converter = MarkitdownConverter(timeout=self.timeout)
        
    async def _arun(self, file_path: str, output_format: str = "markdown") -> ToolResponse:
        """Convert a file to markdown asynchronously."""
        try:
            # Validate input
            if not file_path.strip():
                return ToolResponse(content="Error: Empty file path provided")
            
            # Check if file exists
            if not os.path.exists(file_path):
                return ToolResponse(content=f"Error: File not found: {file_path}")
            
            # Check if it's a file (not directory)
            if not os.path.isfile(file_path):
                return ToolResponse(content=f"Error: Path is not a file: {file_path}")
            
            # Get file info
            file_size = os.path.getsize(file_path)
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Check file size (limit to 100MB for safety)
            max_size = 100 * 1024 * 1024  # 100MB
            if file_size > max_size:
                return ToolResponse(
                    content=f"Error: File too large ({file_size / (1024*1024):.1f}MB). "
                           f"Maximum allowed size is {max_size / (1024*1024)}MB"
                )
            
            # Run conversion in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._convert_file, 
                file_path, 
                output_format
            )
            
            if result is None:
                return ToolResponse(content="Error: Conversion failed - unable to process the file")
            
            # Format the response
            response_content = f"Successfully converted file: {file_name}\n"
            response_content += f"File size: {file_size / 1024:.1f} KB\n"
            response_content += f"File extension: {file_ext}\n"
            response_content += f"Output format: {output_format}\n\n"
            response_content += "--- Converted Content ---\n"
            response_content += result
            
            return ToolResponse(content=response_content)
            
        except asyncio.TimeoutError:
            return ToolResponse(content=f"Error: Conversion timed out after {self.timeout} seconds")
        except Exception as e:
            return ToolResponse(content=f"Error during conversion: {str(e)}")
    
    def _convert_file(self, file_path: str, output_format: str) -> Optional[str]:
        """Convert file to markdown (synchronous helper method)."""
        try:
            result = self.converter.convert(file_path)
            if result and hasattr(result, 'markdown'):
                return result.markdown
            elif isinstance(result, str):
                return result
            else:
                return None
        except Exception as e:
            # Log the error but don't raise it
            logger.error(f"Conversion error: {e}")
            return None
    
    def _run(self, file_path: str, output_format: str = "markdown") -> ToolResponse:
        """Convert a file to markdown synchronously (fallback)."""
        try:
            # Run the async version in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._arun(file_path, output_format))
            finally:
                loop.close()
        except Exception as e:
            return ToolResponse(content=f"Error in synchronous execution: {str(e)}")