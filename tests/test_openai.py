import os
import sys
from pathlib import Path

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from dotenv import load_dotenv
load_dotenv(verbose=True)

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

# Use responses.create() for computer-use-preview model
response = client.responses.create(
    model="computer-use-preview",
    tools=[{
        "type": "computer_use_preview",
        "display_width": 1024,
        "display_height": 768,
        "environment": "browser"  # other possible values: "mac", "windows", "ubuntu"
    }],    
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Check the latest OpenAI news on bing.com."
                }
                # Optional: include a screenshot of the initial state of the environment
                # {
                #     "type": "input_image",
                #     "image_url": f"data:image/png;base64,{screenshot_base64}"
                # }
            ]
        }
    ],
    reasoning={
        "summary": "concise",
    },
    truncation="auto"
)

print(response.output)