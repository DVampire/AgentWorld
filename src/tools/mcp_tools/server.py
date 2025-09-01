import sys
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
import argparse
from mmengine import DictAction, Config
import os
import asyncio

root = str(Path(__file__).resolve().parents[3])
sys.path.append(root)

from src.config import config
from src.logger import logger
from src.models import model_manager
from src.tools.mcp_tools.weather import WeatherTool, WeatherToolArgs, _WEATHER_TOOL_DESCRIPTION

MCP_TOOL_ARGS = {
    "weather": WeatherToolArgs,
}

mcp_server = FastMCP("mcp_server")

def parse_args():
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--config", default=os.path.join(root, "configs", "base.py"), help="config file path")

    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args

def register_weather_tool(config: Config):
    """Register the weather tool."""
    weather_tool = WeatherTool()
    
    mcp_server.tool(
        name="weather",
        title="Weather Tool",
        description=_WEATHER_TOOL_DESCRIPTION,
        annotations=ToolAnnotations(
            category="weather",
            display_name="Weather Tool",
            input_ui={
                "city": {
                    "type": "text",
                    "label": "City",
                    "placeholder": "Enter a city",
                },
            },
        ),
    )(weather_tool._arun)

async def main():
    args = parse_args()
    
    config.init_config(args.config, args)
    logger.init_logger(config)
    logger.info(f"| Config: {config.pretty_text}")
    
    await model_manager.init_models(use_local_proxy=config.use_local_proxy)
    logger.info(f"| Model: {model_manager.list_models()}")
    
    register_weather_tool(config)
    

if __name__ == "__main__":
    asyncio.run(main())
    mcp_server.run(transport="stdio")