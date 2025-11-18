"""Generate an investment-banking style HTML report using a template-based approach."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import requests
from openai import OpenAI
from pydantic import BaseModel, Field

from src.tools.protocol.tool import BaseTool
from src.tools.protocol.types import ToolResponse
from src.tools.protocol import tcp
from src.config import config
from src.utils import assemble_project_path
from src.logger import logger


# --------------------------------------------------------------------------- #
# Environment helpers
# --------------------------------------------------------------------------- #

def load_env() -> None:
    """Load environment variables from a nearby .env if present."""
    for candidate in (
        Path(__file__).resolve().parent / ".env",
        Path.cwd() / ".env",
    ):
        if not candidate.exists():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        break


load_env()

MODEL_NAME = (
    os.getenv("REPORT_ENGINE_MODEL_NAME")
    or os.getenv("OPENAI_MODEL_NAME")
    or os.getenv("DEEPSEEK_MODEL_NAME")
    or "gpt-4o-mini"
)
BASE_URL = (
    os.getenv("REPORT_ENGINE_BASE_URL")
    or os.getenv("OPENAI_API_BASE")
    or os.getenv("OPENAI_BASE_URL")
    or os.getenv("DEEPSEEK_BASE_URL")
)
API_KEY = (
    os.getenv("REPORT_ENGINE_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("DEEPSEEK_API_KEY")
)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# --------------------------------------------------------------------------- #
# Core logic
# --------------------------------------------------------------------------- #

def get_client() -> OpenAI:
    if not API_KEY:
        raise RuntimeError("Missing OpenAI compatible API Key.")
    kwargs: Dict[str, Any] = {"api_key": API_KEY}
    if BASE_URL:
        kwargs["base_url"] = BASE_URL
    return OpenAI(**kwargs)


def search_latest_news(topic: str) -> Dict[str, Any]:
    if not TAVILY_API_KEY:
        raise RuntimeError("Missing TAVILY_API_KEY, unable to search news.")
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": topic,
        "search_depth": "advanced",
        "include_answer": True,
        "max_results": 10,
        "time_range": "week",
    }
    resp = requests.post("https://api.tavily.com/search", json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return {
        "query": topic,
        "answer": data.get("answer"),
        "results": [
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "content": item.get("content") or item.get("snippet"),
                "published_date": item.get("published_date"),
            }
            for item in data.get("results", [])
        ],
    }


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_latest_news",
            "description": "Search for latest news data related to the topic, focusing on the last 7 days by default, for report writing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic or question description to search",
                    }
                },
                "required": ["topic"],
            },
        },
    }
]

TOOL_HANDLERS = {
    "search_latest_news": search_latest_news,
}


JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_from_response(raw: str) -> Dict[str, Any]:
    """Extract JSON data from model output."""
    # Try to extract from code block
    match = JSON_BLOCK_PATTERN.search(raw)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try to find JSON object directly
    json_start = raw.find("{")
    if json_start != -1:
        json_end = raw.rfind("}") + 1
        if json_end > json_start:
            try:
                return json.loads(raw[json_start:json_end])
            except json.JSONDecodeError:
                pass

    # If all fail, raise error
    raise RuntimeError(f"Unable to extract valid JSON data from model output. First 200 characters: {raw[:200]}")


def generate_report_data(query: str) -> Dict[str, Any]:
    """Generate report data (JSON format), including relationship graph, upside/downside probabilities and reasons."""
    client = get_client()
    system_prompt = (
        "You are a professional equity research analyst. Extract entity relationships from news and analyze stock price movement probabilities. "
        "Based on the provided news data, generate a JSON-formatted analysis report containing the following:\n\n"
        "1. **Relationship Graph Data** (relationshipGraph):\n"
        "   - Extract entities and relationships from news sentiment analysis to reveal market dynamics\n"
        "   - Focus on market-relevant entities: companies, key personnel, products, events, regulatory bodies, competitors, partners, etc.\n"
        "   - Extract relationships that reflect market sentiment, business connections, competitive dynamics, and strategic moves\n"
        "   - nodes: Array of nodes, each containing {id, label, title(optional), color(optional), shape(optional)}\n"
        "   - edges: Array of edges, each containing {from, to, label(relationship description)}\n"
        "   - Extract at least 5-10 relevant entities and their relationships that help understand market dynamics\n"
        "   - IMPORTANT: Keep node labels SHORT (max 15 characters, e.g., 'Apple Inc.' not 'Apple Inc. (AAPL) Technology Company')\n"
        "   - Use concise edge labels (max 3-4 words, e.g., 'leads', 'develops', 'competes with', 'invests in')\n"
        "   - Edge labels should be clear and descriptive but brief (they will be displayed on the graph)\n"
        "   - Put detailed descriptions in the 'title' field for tooltips\n"
        "   - **Color coding for nodes (IMPORTANT):**\n"
        "     * Companies: Use blue colors (e.g., '#6366f1', '#3b82f6', '#2563eb', '#0a84ff')\n"
        "     * People: Use orange/yellow colors (e.g., '#f59e0b', '#f97316', '#ff9500', '#ffcc00')\n"
        "     * Products: Use green colors (e.g., '#10b981', '#34c759', '#00c853', '#34d399')\n"
        "     * Events: Use purple colors (e.g., '#8b5cf6', '#7c3aed', '#5856d6', '#a78bfa')\n"
        "     * Other entities: Use gray colors (e.g., '#64748b', '#8e8e93', '#a3a3a8')\n"
        "   - All labels and descriptions must be in English\n\n"
        "2. **Upside Probability Analysis** (upProbability, upReasonSummary, upReasonDetail):\n"
        "   - upProbability: Upside probability (integer 0-100)\n"
        "   - upReasonSummary: Brief summary of upside reasons (1-2 sentences, in English)\n"
        "   - upReasonDetail: Detailed analysis of upside reasons (3-5 paragraphs, 2-3 sentences each, in English)\n\n"
        "3. **Downside Probability Analysis** (downProbability, downReasonSummary, downReasonDetail):\n"
        "   - downProbability: Downside probability (integer 0-100)\n"
        "   - downReasonSummary: Brief summary of downside reasons (1-2 sentences, in English)\n"
        "   - downReasonDetail: Detailed analysis of downside reasons (3-5 paragraphs, 2-3 sentences each, in English)\n\n"
        "4. **News Analysis** (newsAnalysisSummary):\n"
        "   - newsAnalysisSummary: Comprehensive news analysis covering major events, announcements, regulatory changes, product launches, and their potential impact (3-5 paragraphs, 2-3 sentences each, in English)\n\n"
        "5. **Relationship Graph Interpretation** (graphInterpretationSummary):\n"
        "   - graphInterpretationSummary: Comprehensive interpretation of the relationship graph including entity connections, relationship patterns, key influencers, and network insights (3-5 paragraphs, 2-3 sentences each, in English)\n\n"
        "6. **Metadata** (companyName, reportDate):\n"
        "   - companyName: Company name (in English)\n"
        "   - reportDate: Report date (format: YYYY-MM-DD). MUST use TODAY's date in YYYY-MM-DD format. Do NOT use example dates like '2024-01-01'.\n\n"
        "Output strictly in the following JSON format, without any additional text:\n"
        "{\n"
        '  "companyName": "Company Name",\n'
        '  "reportDate": "TODAY\'S DATE IN YYYY-MM-DD FORMAT",\n'
        '  "relationshipGraph": {\n'
        '    "nodes": [{"id": "1", "label": "Entity 1", "title": "Detailed description", "color": "#667eea"}],\n'
        '    "edges": [{"from": "1", "to": "2", "label": "Relationship description"}]\n'
        '  },\n'
        '  "upProbability": 65,\n'
        '  "upReasonSummary": "Brief summary of upside reasons",\n'
        '  "upReasonDetail": "Detailed analysis of upside reasons, can be multiple paragraphs.",\n'
        '  "downProbability": 35,\n'
        '  "downReasonSummary": "Brief summary of downside reasons",\n'
        '  "downReasonDetail": "Detailed analysis of downside reasons, can be multiple paragraphs.",\n'
        '  "newsAnalysisSummary": "Comprehensive news analysis covering major events, announcements, and their potential impact.",\n'
        '  "graphInterpretationSummary": "Comprehensive interpretation of the relationship graph including entity connections and network insights."\n'
        "}\n\n"
        "IMPORTANT: All text content (labels, summaries, details) must be in English. "
        "The sum of upProbability and downProbability should be close to 100 (small deviations allowed)."
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    while True:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
        )
        message = response.choices[0].message

        if not message.tool_calls:
            if not message.content:
                raise RuntimeError("Model did not return any content.")
            data = extract_json_from_response(message.content)
            # Automatically correct report date to current date
            today = datetime.now().strftime("%Y-%m-%d")
            if data.get("reportDate") != today:
                logger.info(f"⚠️  Correcting report date from '{data.get('reportDate')}' to '{today}'")
                data["reportDate"] = today
            return data

        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": message.tool_calls,
            }
        )

        for tool_call in message.tool_calls:
            handler = TOOL_HANDLERS.get(tool_call.function.name)
            if handler is None:
                continue
            args = json.loads(tool_call.function.arguments or "{}")
            topic = args.get("topic", query)
            result = handler(topic)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )


# --------------------------------------------------------------------------- #
# Template and rendering
# --------------------------------------------------------------------------- #

def load_template() -> str:
    """Load HTML template."""
    template_path = Path(__file__).parent / "report_template.html"
    if not template_path.exists():
        raise RuntimeError(f"Template file not found: {template_path}")
    return template_path.read_text(encoding="utf-8")


def render_report(data: Dict[str, Any]) -> str:
    """Fill data into template to generate final HTML report."""
    template = load_template()
    
    # Convert data to JSON string and embed in HTML
    data_json = json.dumps(data, ensure_ascii=False, indent=2)
    
    # Replace content in data script tag
    script_tag = f'<script id="report-data">\n        window.REPORT_DATA = {data_json};\n    </script>'
    
    # Replace data script in template (match English comment)
    old_script = '<script id="report-data">\n        // Data will be populated during rendering\n        window.REPORT_DATA = window.REPORT_DATA || {};\n    </script>'
    
    if old_script in template:
        html = template.replace(old_script, script_tag)
    else:
        # If not found, try more flexible matching
        pattern = r'<script id="report-data">.*?window\.REPORT_DATA = window\.REPORT_DATA \|\| \{\};.*?</script>'
        html = re.sub(pattern, script_tag, template, flags=re.DOTALL)
    
    return html


# --------------------------------------------------------------------------- #
# Tool Definition
# --------------------------------------------------------------------------- #

_REPORT_TOOL_DESCRIPTION = """Generate investment-banking style HTML research reports using a template-based approach.

Use this tool to generate professional HTML format research reports based on a given topic. The report includes:
- Market Relationship Graph (interactive visualization)
- News Analysis & Graph Interpretation
- Stock Price Movement Probability Analysis (Upside/Downside)
- Comprehensive analysis of market dynamics

The report automatically searches for the latest news data and generates a professional HTML document with interactive graphs, probability analysis, and complete market analysis.
"""


class ReportToolArgs(BaseModel):
    query: str = Field(description="Topic or question description for generating the report")
    output_path: Optional[str] = Field(
        default=None,
        description="Optional: Specify the full path (including filename) to save the report. If not specified, saves to default location (workdir/reports/)"
    )


@tcp.tool()
class ReportTool(BaseTool):
    """A tool for generating investment-banking style HTML reports using template-based approach."""
    
    name: str = "report"
    type: str = "Report Generation"
    description: str = _REPORT_TOOL_DESCRIPTION
    args_schema: Type[BaseModel] = ReportToolArgs
    metadata: Dict[str, Any] = {}
    
    def __init__(self, base_dir: Optional[str] = None, **kwargs):
        """Initialize the report tool."""
        super().__init__(**kwargs)
        # If base_dir is specified, use it; otherwise get workdir from config
        if base_dir:
            self.base_dir = assemble_project_path(base_dir)
        else:
            # Try to get workdir from config, if not available use default value
            try:
                workdir = config.get("workdir", "workdir/reports")
                self.base_dir = assemble_project_path(workdir)
            except Exception:
                # If config is not initialized, use default value
                self.base_dir = Path.cwd() / "workdir" / "reports"
        
        # Ensure directory exists
        Path(self.base_dir).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Report tool initialized with base_dir: {self.base_dir}")
    
    def _save_report(self, html: str, query: str, output_path: Optional[str] = None) -> Path:
        """Save report to specified path or default location"""
        if output_path:
            # User specified full path
            file_path = Path(assemble_project_path(output_path))
            file_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # Use default location: base_dir/reports/
            output_dir = Path(self.base_dir) / "reports"
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_query = "".join(c for c in query if c.isalnum() or c in (" ", "-", "_")).strip()
            safe_query = safe_query.replace(" ", "_")[:50] or "report"
            file_path = output_dir / f"tool_report_{safe_query}_{timestamp}.html"
        
        file_path.write_text(html, encoding="utf-8")
        return file_path
    
    async def _arun(self, query: str, output_path: Optional[str] = None) -> ToolResponse:
        """Asynchronously execute report generation"""
        try:
            logger.info(f"🔍 Starting report generation: {query}")
            
            # Generate report data (JSON format)
            logger.info("📊 Generating report data...")
            data = generate_report_data(query)
            logger.info("✅ Report data generated successfully")
            
            # Render HTML
            logger.info("📝 Rendering HTML report from template...")
            html = render_report(data)
            
            # Save report
            path = self._save_report(html, query, output_path)
            logger.info(f"✅ Report generated: {path}")
            
            return ToolResponse(
                success=True,
                message=f"Report successfully generated and saved to: {path}\nFile size: {len(html)} characters",
                extra={
                    "path": str(path),
                    "absolute_path": str(path.resolve()),
                    "html_length": len(html),
                    "query": query
                }
            )
        except Exception as e:
            logger.error(f"❌ Failed to generate report: {str(e)}")
            return ToolResponse(
                success=False,
                message=f"Failed to generate report: {str(e)}",
                extra={"error_type": type(e).__name__, "query": query}
            )
    
    def _run(self, query: str, output_path: Optional[str] = None) -> ToolResponse:
        """Synchronously execute report generation (fallback)"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._arun(query, output_path))
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Failed to synchronously execute report generation: {str(e)}")
            return ToolResponse(
                success=False,
                message=f"Synchronous execution failed: {str(e)}",
                extra={"error_type": type(e).__name__}
            )


# --------------------------------------------------------------------------- #
# Standalone execution (for backward compatibility)
# --------------------------------------------------------------------------- #

def save_report(html: str, query: str) -> Path:
    """Save report (helper function for standalone execution)"""
    output_dir = Path("final_reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = "".join(c for c in query if c.isalnum() or c in (" ", "-", "_")).strip()
    safe_query = safe_query.replace(" ", "_")[:50] or "report"
    file_path = output_dir / f"tool_report_{safe_query}_{timestamp}.html"
    file_path.write_text(html, encoding="utf-8")
    return file_path


def main() -> None:
    """Standalone execution entry point (for testing)"""
    if len(sys.argv) < 2:
        print('Usage: python report.py "What is the latest news about Apple stock price?"')
        sys.exit(1)

    query = " ".join(sys.argv[1:]).strip()
    print(f"Generating report data: {query}")
    
    # Generate report data
    data = generate_report_data(query)
    print("Report data generated, rendering HTML...")
    
    # Render HTML
    html = render_report(data)
    
    # Save report
    path = save_report(html, query)
    print(f"Report generated: {path}")


if __name__ == "__main__":
    main()
