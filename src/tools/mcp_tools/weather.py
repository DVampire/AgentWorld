
from mcp.types import ToolAnnotations
from mcp.server.fastmcp.server import Context
from pydantic import BaseModel, Field

from src.tools.mcp_tools.server import mcp_server

_WEATHER_TOOL_DESCRIPTION = """
Get the weather of a city.
"""

class WeatherToolInput(BaseModel):
    city: str = Field(description="The city to get the weather of.")

class WeatherToolOutput(BaseModel):
    weather: str = Field(description="The weather of the city.")

@mcp_server.tool(
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
            }
        },
    ),  
)
async def get_weather(args: WeatherToolInput, ctx: Context) -> WeatherToolOutput:
    """
    Get the weather of a city.

    Args:
        weather_input: The input of the weather tool.

    Returns:
        dict: The weather of the city.
    """
    await ctx.info(f"Getting weather for {args.city}")
    res: WeatherToolOutput = WeatherToolOutput(weather="sunny")
    return res