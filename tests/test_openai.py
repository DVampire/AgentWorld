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

tools = [
    {
        "name": "create_user",
        "type": "function",
        "description": "创建用户对象",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"],
            "additionalProperties": False
        }
    }
]

text = """
你现在需要找到一个用户，名字叫 Alice，年龄 25 岁，请使用工具 create_user 创建一个用户对象。
<output>
You must ALWAYS respond with a valid JSON in this exact format:

{
  "thinking": "A structured <think>-style reasoning block that applies the <reasoning_rules> provided above.",
  "evaluation_previous_goal": "One-sentence analysis of your last action. Clearly state success, failure, or uncertain.",
  "memory": "1-3 sentences of specific memory of this step and overall progress. You should put here everything that will help you track progress in future steps.",
  "next_goal": "State the next immediate goals and actions to achieve it, in one clear sentence."
  "action": [{"one_action_name": {// action-specific parameters}}, // ... more actions in sequence]
}

Action list should NEVER be empty.
</output>
"""



# 调用 Responses API
response = client.responses.create(
    model="gpt-5",
    input=text,
    tools=tools,
)

print(response)