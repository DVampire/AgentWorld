"""Plotter Tool - A workflow agent for generating plots from text, markdown tables, or CSV files."""

import os
import re
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict

from src.utils import dedent
from src.registry import TOOL
from src.logger import logger
from src.model import model_manager
from src.utils import assemble_project_path
from src.tool.types import Tool, ToolResponse
from src.message import HumanMessage, SystemMessage
from src.tool.default_tools.python_interpreter import PythonInterpreterTool

class CodeGeneration(BaseModel):
    """Generated Python code for data conversion or plotting."""
    code: str = Field(description="The Python code to execute")


_PLOTTER_DESCRIPTION = """Plotter tool that generates visualizations from text, markdown tables, or CSV files.

🎯 BEST FOR: Creating data visualizations from various input formats:
- Plain text data
- Markdown table strings
- CSV files

This tool will:
1. Detect the input format (text, markdown table, or CSV file)
2. If text or markdown table: Generate Python code to convert to CSV
3. Execute the conversion code using Python interpreter
4. Generate seaborn plotting code based on the data
5. Execute the plotting code to create a PNG visualization
6. Save the plot as a PNG file

💡 Use this tool for:
- Visualizing data from text descriptions
- Converting markdown tables to charts
- Creating plots from CSV data files
- Generating data visualizations automatically
"""


@TOOL.register_module(force=True)
class PlotterTool(Tool):
    """A tool that generates plots from text, markdown tables, or CSV files."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = "plotter"
    description: str = _PLOTTER_DESCRIPTION
    enabled: bool = True

    # Configuration parameters
    model_name: str = Field(
        default="openrouter/gemini-3-flash-preview",
        description="The model to use for code generation."
    )
    base_dir: str = Field(
        default=None,
        description="The base directory to use for saving files."
    )
    python_interpreter: Optional[PythonInterpreterTool] = Field(
        default=None,
        description="The Python interpreter tool to use."
    )

    def __init__(self, model_name: Optional[str] = None, base_dir: Optional[str] = None, **kwargs):
        """Initialize the plotter tool."""
        super().__init__(**kwargs)

        if model_name is not None:
            self.model_name = model_name
        
        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
            
        if self.base_dir is not None:
            os.makedirs(self.base_dir, exist_ok=True)

        # Initialize Python interpreter tool
        if self.python_interpreter is None:
            self.python_interpreter = PythonInterpreterTool()

    def _is_csv_file(self, input_data: str) -> bool:
        """Check if input is a CSV file path."""
        # Check if it's a file path that ends with .csv
        if os.path.exists(input_data) and input_data.lower().endswith('.csv'):
            return True
        return False

    def _is_markdown_table(self, input_data: str) -> bool:
        """Check if input is a markdown table string."""
        # Markdown tables typically have | separators and - separators
        lines = input_data.strip().split('\n')
        if len(lines) < 2:
            return False
        
        # Check for table pattern: lines with | and a separator line with - or =
        has_pipe = any('|' in line for line in lines)
        has_separator = any(re.match(r'^[\s\|:\-\=]+$', line) for line in lines)
        
        return has_pipe and has_separator

    def _is_text_data(self, input_data: str) -> bool:
        """Check if input is plain text data (not a file path or markdown table)."""
        # If it's a file path, it's not text data
        if os.path.exists(input_data):
            return False
        
        # If it's a markdown table, it's not plain text
        if self._is_markdown_table(input_data):
            return False
        
        # Otherwise, treat as text
        return True

    async def _generate_csv_conversion_code(self, input_data: str, output_csv_path: str) -> str:
        """Generate Python code to convert text or markdown table to CSV."""
        input_type = "markdown table" if self._is_markdown_table(input_data) else "text"
        
        prompt = dedent(f"""Generate Python code to convert the following {input_type} data into a CSV file.

        Input data:
        {input_data}

        Requirements:
        1. Parse the {input_type} and extract structured data
        2. If it's a markdown table, parse the table structure (headers and rows)
        3. If it's plain text, identify the data structure (e.g., key-value pairs, lists, etc.) and convert to tabular format
        4. Convert it to a pandas DataFrame with appropriate column names
        5. Save the DataFrame to CSV file at: {output_csv_path}
        6. Use pandas.to_csv() method
        7. Include error handling if the data format is unexpected
        8. Print a success message when the CSV is saved, including the file path

        Important: The code must be complete and executable. Include all necessary imports (pandas, re, etc.).
        Return ONLY the Python code, without any markdown code blocks or explanations.
        The code should be ready to execute directly.
        """)

        messages = [
            SystemMessage(content="You are an expert at writing Python code for data conversion. Generate clean, executable Python code."),
            HumanMessage(content=prompt)
        ]

        response = await model_manager(
            model=self.model_name,
            messages=messages,
            response_format=CodeGeneration
        )

        if response.extra and "parsed_model" in response.extra:
            code_generation = response.extra["parsed_model"]
            return code_generation.code
        else:
            # Fallback: extract code from message
            code = response.message.strip()
            # Remove markdown code blocks if present
            code = re.sub(r'```python\n?', '', code)
            code = re.sub(r'```\n?', '', code)
            return code.strip()

    async def _generate_plotting_code(self, csv_path: str, output_png_path: str) -> str:
        """Generate Python code to create a plot from CSV using seaborn."""
        # First, read a sample of the CSV to understand the data structure
        columns = []
        sample_data = ""
        try:
            # Read first few lines to understand structure
            with open(csv_path, 'r', encoding='utf-8') as f:
                lines = [f.readline().strip() for _ in range(6)]  # Header + 5 data rows
            
            if lines:
                # Parse header
                header = lines[0]
                columns = [col.strip() for col in header.split(',')]
                # Get sample data
                sample_data = '\n'.join(lines[:6])
        except Exception as e:
            logger.warning(f"Could not read CSV sample: {e}")
            columns = []
            sample_data = "Unable to read CSV file"

        prompt = dedent(f"""Generate Python code to create a visualization from a CSV file using seaborn.

        CSV file path: {csv_path}
        Output PNG path: {output_png_path}

        CSV columns: {columns}
        Sample data (first 5 rows):
        {sample_data}

        Requirements:
        1. Read the CSV file using pandas: pd.read_csv()
        2. Analyze the data structure (columns, data types, number of rows) and choose an appropriate plot type:
           - For categorical data: bar plot, count plot, or box plot
           - For numerical relationships: scatter plot, line plot, or regression plot
           - For time series: line plot
           - For correlations: heatmap
           - For distributions: histogram, kde plot, or violin plot
        3. Create a beautiful and informative visualization using seaborn (sns)
        4. Add proper labels (xlabel, ylabel), title, and formatting
        5. Adjust figure size if needed for better readability
        6. Save the plot as a PNG file using: plt.savefig('{output_png_path}', dpi=100, bbox_inches='tight')
        7. Close the figure after saving: plt.close()
        8. Include error handling with try-except
        9. Print a success message when the plot is saved, including the file path

        Important: The code must be complete and executable. Include all necessary imports:
        - import pandas as pd
        - import seaborn as sns
        - import matplotlib.pyplot as plt

        Return ONLY the Python code, without any markdown code blocks or explanations.
        The code should be ready to execute directly.
        """)

        messages = [
            SystemMessage(content="You are an expert at creating data visualizations with seaborn. Generate clean, executable Python code that creates informative and visually appealing plots."),
            HumanMessage(content=prompt)
        ]

        response = await model_manager(
            model=self.model_name,
            messages=messages,
            response_format=CodeGeneration
        )

        if response.extra and "parsed_model" in response.extra:
            code_generation = response.extra["parsed_model"]
            return code_generation.code
        else:
            # Fallback: extract code from message
            code = response.message.strip()
            # Remove markdown code blocks if present
            code = re.sub(r'```python\n?', '', code)
            code = re.sub(r'```\n?', '', code)
            return code.strip()

    async def __call__(self, input_data: str, output_filename: Optional[str] = None, **kwargs) -> ToolResponse:
        """Execute plotter workflow.

        Args:
            input_data (str): The input data - can be: Plain text data, Markdown table string, or Path to a CSV file
            output_filename (Optional[str]): Optional custom filename for the output PNG. If not provided, a default name will be generated.
        """
        try:
            logger.info(f"| 🚀 Starting PlotterTool with input: {input_data[:100]}...")

            # Determine input type
            if self._is_csv_file(input_data):
                logger.info(f"| 📊 Input detected as CSV file: {input_data}")
                csv_path = os.path.abspath(input_data)
                needs_conversion = False
            elif self._is_markdown_table(input_data):
                logger.info(f"| 📋 Input detected as markdown table")
                needs_conversion = True
            elif self._is_text_data(input_data):
                logger.info(f"| 📝 Input detected as text data")
                needs_conversion = True
            else:
                return ToolResponse(
                    success=False,
                    message=f"Unable to determine input type. Please provide text, markdown table, or CSV file path."
                )

            # Step 1: Convert to CSV if needed
            if needs_conversion:
                logger.info(f"| 🔄 Step 1: Converting input to CSV...")
                
                # Generate output CSV path
                import uuid
                csv_filename = f"plotter_data_{uuid.uuid4().hex[:8]}.csv"
                csv_path = os.path.abspath(os.path.join(self.base_dir, csv_filename))
                
                # Generate conversion code
                conversion_code = await self._generate_csv_conversion_code(input_data, csv_path)
                logger.info(f"| 📝 Generated conversion code:\n{conversion_code[:200]}...")
                
                # Execute conversion code
                logger.info(f"| ⚙️ Executing conversion code...")
                conversion_result = await self.python_interpreter(code=conversion_code)
                
                if not conversion_result.success:
                    return ToolResponse(
                        success=False,
                        message=f"Failed to convert input to CSV: {conversion_result.message}"
                    )
                
                logger.info(f"| ✅ CSV conversion successful: {csv_path}")
                logger.info(f"| 📄 Conversion output: {conversion_result.message[:200]}...")
                
                # Verify CSV file was created
                if not os.path.exists(csv_path):
                    return ToolResponse(
                        success=False,
                        message=f"CSV file was not created at {csv_path}. Conversion code may have failed silently."
                    )
            else:
                # Verify CSV file exists
                if not os.path.exists(csv_path):
                    return ToolResponse(
                        success=False,
                        message=f"CSV file not found: {csv_path}"
                    )

            # Step 2: Generate plotting code
            logger.info(f"| 🎨 Step 2: Generating plotting code...")
            
            # Generate output PNG path
            if output_filename:
                if not output_filename.endswith('.png'):
                    output_filename += '.png'
                png_path = os.path.abspath(os.path.join(self.base_dir, output_filename))
            else:
                import uuid
                png_filename = f"plot_{uuid.uuid4().hex[:8]}.png"
                png_path = os.path.abspath(os.path.join(self.base_dir, png_filename))
            
            # Generate plotting code
            plotting_code = await self._generate_plotting_code(csv_path, png_path)
            logger.info(f"| 📝 Generated plotting code:\n{plotting_code[:200]}...")
            
            # Execute plotting code
            logger.info(f"| ⚙️ Executing plotting code...")
            plotting_result = await self.python_interpreter(code=plotting_code)
            
            if not plotting_result.success:
                return ToolResponse(
                    success=False,
                    message=f"Failed to generate plot: {plotting_result.message}"
                )
            
            logger.info(f"| ✅ Plot generation successful: {png_path}")
            logger.info(f"| 📄 Plotting output: {plotting_result.message[:200]}...")
            
            # Verify PNG file was created
            if not os.path.exists(png_path):
                return ToolResponse(
                    success=False,
                    message=f"PNG file was not created at {png_path}. Plotting code may have failed silently."
                )
            
            # Return success with absolute file paths
            csv_abs_path = os.path.abspath(csv_path)
            png_abs_path = os.path.abspath(png_path)
            return ToolResponse(
                success=True,
                message=f"Plot successfully generated!\n\nCSV file (absolute path): {csv_abs_path}\nPNG file (absolute path): {png_abs_path}",
                extra={"csv_path": csv_abs_path, "png_path": png_abs_path}
            )

        except Exception as e:
            logger.error(f"| ❌ Error in plotter: {e}")
            import traceback
            return ToolResponse(
                success=False,
                message=f"Error during plot generation: {str(e)}\n{traceback.format_exc()}"
            )
