import sys
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
import argparse
from mmengine import DictAction, Config
import os
import asyncio
import aiohttp

root = str(Path(__file__).resolve().parents[3])
sys.path.append(root)

from src.config import config
from src.logger import logger
from src.infrastructures import model_manager

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

@mcp_server.tool(
    name="weather",
    title="Weather Tool",
    description="Get the weather of a city.",
)
async def weather(city: str) -> str:
    """Get the weather of a city.
    """
    
    geocode_url = "https://geocoding-api.open-meteo.com/v1/search"
    weather_url = "https://geocoding-api.open-meteo.com/v1/search"
    
    try:
        # Step 1: geocode city -> lat/lon
        async with aiohttp.ClientSession() as session:
            async with session.get(geocode_url, params={"name": city}) as resp:
                if resp.status != 200:
                    return f"Error fetching geocode for {city}: HTTP {resp.status}"
                geo = await resp.json()
                if not geo.get("results"):
                    return f"Could not find location for city: {city}"
                lat = geo["results"][0]["latitude"]
                lon = geo["results"][0]["longitude"]

            # Step 2: fetch weather
            params = {"latitude": lat, "longitude": lon, "current_weather": "true"}
            async with session.get(weather_url, params=params) as resp:
                if resp.status != 200:
                    return f"Error fetching weather for {city}: HTTP {resp.status}"
                data = await resp.json()
                cw = data.get("current_weather", {})
                temp = cw.get("temperature")
                wind = cw.get("windspeed")
                return (
                    f"Weather in {city}:\n"
                    f"- Temperature: {temp}°C\n"
                    f"- Wind speed: {wind} m/s"
                )
    except Exception as e:
        return f"Error in WeatherTool: {e}"

async def main():
    args = parse_args()
    
    config.init_config(args.config, args)
    logger.init_logger(config)
    logger.info(f"| Config: {config.pretty_text}")
    
    await model_manager.initialize(use_local_proxy=config.use_local_proxy)
    logger.info(f"| Model: {model_manager.list()}")

if __name__ == "__main__":
    # Initialize the server first
    asyncio.run(main())
    # Then run the MCP server
    mcp_server.run(transport="stdio")