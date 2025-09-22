"""Simple test for OpenAI Computer Use API browser."""

import os
import sys
from dotenv import load_dotenv
load_dotenv(verbose=True)

from pathlib import Path
import argparse
from mmengine import DictAction
import asyncio

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.config import config
from src.logger import logger
from src.infrastructures.models import model_manager
from src.environments.browser.openai_browser import OpenAIBrowser

def parse_args():
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--config", default=os.path.join(root, "configs", "tool_calling_agent.py"), help="config file path")

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


def convert_computer_use_action(action):
    """Convert Computer Use API action to our browser action format."""
    action_dict = {
        "type": action.type
    }
    
    if action.type == "click":
        action_dict.update({
            "x": action.x,
            "y": action.y,
            "button": action.button
        })
    elif action.type == "double_click":
        action_dict.update({
            "x": action.x,
            "y": action.y,
            "button": action.button
        })
    elif action.type == "type":
        action_dict.update({
            "text": action.text
        })
    elif action.type == "keypress":
        action_dict.update({
            "keys": action.keys
        })
    elif action.type == "scroll":
        action_dict.update({
            "x": action.x,
            "y": action.y,
            "scroll_x": action.scroll_x,
            "scroll_y": action.scroll_y
        })
    elif action.type == "wait":
        action_dict.update({
            "ms": getattr(action, 'ms', 2000)
        })
    elif action.type == "move":
        action_dict.update({
            "x": action.x,
            "y": action.y
        })
    elif action.type == "drag":
        action_dict.update({
            "path": action.path
        })
    elif action.type == "screenshot":
        action_dict.update({
            "full_page": getattr(action, 'full_page', False)
        })
    
    return action_dict


async def test_computer_use_workflow():
    """Test Computer Use API workflow with model-generated actions."""
    print("=== Testing OpenAI Computer Use API with Model ===\n")
    
    browser = OpenAIBrowser(headless=False)
    model = model_manager.get("computer-browser-use")
    
    # Screenshot counter for naming
    screenshot_counter = 0
    
    def save_screenshot(screenshot_data, description=""):
        nonlocal screenshot_counter
        screenshot_counter += 1
        filename = f"screenshot_{screenshot_counter:03d}_{description}.png"
        with open(filename, "wb") as f:
            import base64
            f.write(base64.b64decode(screenshot_data))
        print(f"📸 Screenshot saved: {filename}")
        return filename
    
    try:
        await browser.start()
        print("✓ Browser started")
        
        # Initial navigation
        nav_action = {
            "type": "navigate",
            "url": "https://www.google.com"
        }
        result = await browser.execute(nav_action)
        print(f"✓ Initial navigation: {result['success']}")
        
        # Define the task for the model
        task = "Search for 'OpenAI Computer Use API' on Google"
        print(f"\n📋 Task: {task}")
        
        # Get initial screenshot
        screenshot = await browser.take_screenshot()
        save_screenshot(screenshot, "initial_google")
        
        # Create messages for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Please help me complete this task: {task}"
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{screenshot}"
                    }
                ]
            }
        ]
        
        print("🤖 Calling Computer Use model...")
        
        # Call the model to get actions
        response = await model.ainvoke(
            messages,
            reasoning={
                "generate_summary": "concise",
            },
        )
        print(f"✓ Model response received")
        
        # Process the response and execute actions
        if hasattr(response, 'output') and response.output:
            print(f"🔧 Found {len(response.output)} actions to execute")
            
            for i, output_item in enumerate(response.output, 1):
                if hasattr(output_item, 'action') and output_item.action:
                    action = output_item.action
                    print(f"\n--- Executing Action {i}: {action.type} ---")
                    
                    # Convert Computer Use API action to our browser action format
                    browser_action = convert_computer_use_action(action)
                    print(f"  Converted action: {browser_action}")
                    
                    result = await browser.execute(browser_action)
                    print(f"✓ Action {i}: {result['success']} - {result.get('message', '')}")
                    
                    if result.get('screenshot'):
                        print(f"  Screenshot: {len(result['screenshot'])} chars")
                        save_screenshot(result['screenshot'], f"action_{i}_{action.type}")
                elif hasattr(output_item, 'summary'):
                    print(f"📝 Reasoning: {output_item.summary}")
        
        # Take final screenshot
        final_screenshot = await browser.take_screenshot()
        print(f"\n✓ Final screenshot: {len(final_screenshot)} chars")
        save_screenshot(final_screenshot, "final_result")
                
        print("\n🎉 Computer Use workflow completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await browser.stop()
        print("✓ Browser stopped")


async def main():
    args = parse_args()
    
    config.init_config(args.config, args)
    logger.init_logger(config)
    logger.info(f"| Config: {config.pretty_text}")
    
    # Initialize model manager
    logger.info("| 🧠 Initializing model manager...")
    await model_manager.initialize(use_local_proxy=config.use_local_proxy)
    logger.info(f"| ✅ Model manager initialized: {model_manager.list()}")
    
    await test_computer_use_workflow()
    
    logger.info("| 🚪 Test completed")

if __name__ == "__main__":
    asyncio.run(main())
