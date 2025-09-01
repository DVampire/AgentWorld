import aiohttp
from pydantic import BaseModel, Field
from typing import Dict, Any, Type

_WEATHER_TOOL_DESCRIPTION = """
Use this tool to fetch the current weather of a city using Open-Meteo (free, no API key).
"""

class WeatherToolArgs(BaseModel):
    city: str = Field(description="The city to get weather for.")

class WeatherTool:
    
    name: str = "weather"
    description: str = _WEATHER_TOOL_DESCRIPTION
    args_schema: Type[WeatherToolArgs] = WeatherToolArgs
    
    def __init__(self, **kwargs):
        
        self.geocode_url = "https://geocoding-api.open-meteo.com/v1/search"
        self.weather_url = "https://api.open-meteo.com/v1/forecast"
        
        super().__init__(**kwargs)

    async def _arun(self, city: str) -> str:
        try:
            # Step 1: geocode city -> lat/lon
            async with aiohttp.ClientSession() as session:
                async with session.get(self.geocode_url, params={"name": city}) as resp:
                    if resp.status != 200:
                        return f"Error fetching geocode for {city}: HTTP {resp.status}"
                    geo = await resp.json()
                    if not geo.get("results"):
                        return f"Could not find location for city: {city}"
                    lat = geo["results"][0]["latitude"]
                    lon = geo["results"][0]["longitude"]

                # Step 2: fetch weather
                params = {"latitude": lat, "longitude": lon, "current_weather": "true"}
                async with session.get(self.weather_url, params=params) as resp:
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
        
    def _run(self, city: str) -> str:
        """Fetch the current weather of a city."""
        try:
            return self._arun(city)
        except Exception as e:
            return f"Error in WeatherTool: {str(e)}"
    
    def get_tool_config(self) -> Dict[str, Any]:
        """Get the tool configuration."""
        return {
            "name": self.name,
            "description": self.description,
            "args_schema": self.args_schema
        }
