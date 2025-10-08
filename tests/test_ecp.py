"""Example of running the InteractiveAgent with Cursor-style interaction."""

import os
import sys
import time
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
from src.environments import ecp
from src.tools import tcp
from src.utils import assemble_project_path

def parse_args():
        parser = argparse.ArgumentParser(description='Tool Calling Agent Example')
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

async def test_file_system():
    
    input = {
        "name": "file_system",
        "action": "read",
        "input": {
            "file_path": assemble_project_path("tests/files/test.txt")
        }
    }
    
    res = await ecp.ainvoke(**input)
    logger.info(f"| ✅ Action result: {res}")
    
async def test_operator_browser():
    
    res = await ecp.ainvoke(
        name="operator_browser",
        action="type",
        input={
            "text": "www.google.com"
        }
    )
    print(res)
    
    state = await ecp.get_state("operator_browser")
    
async def test_github():
    """Test GitHub environment workflow with proper Git workflow"""
    
    print("🚀 Starting GitHub Environment Workflow Test")
    print("=" * 50)
    
    # Step 1: Create a new GitHub repository (remote)
    print("\n📝 Step 1: Create a new GitHub repository")
    repo_name = f"ecp-test-{int(time.time())}"  # Unique repo name
    res1 = await ecp.ainvoke(
        name="github",
        action="create_repository",
        input={
            "name": repo_name,
            "description": "Test repository created by ECP workflow",
            "private": False,
            "auto_init": True
        }
    )
    print(f"Result: {res1}")
    
    # Step 2: Clone the remote repository to local
    print("\n📥 Step 2: Clone the remote repository to local")
    res2 = await ecp.ainvoke(
        name="github",
        action="git_clone",
        input={
            "owner": os.getenv("GITHUB_USERNAME"),  # Replace with actual GitHub username
            "repo": repo_name,
            "local_path": assemble_project_path(f"workdir/{repo_name}"),
            "branch": "main"
        }
    )
    print(f"Result: {res2}")
    
    # Step 3: Create a test file using file system environment
    print("\n📄 Step 3: Create a test file")
    res3 = await ecp.ainvoke(
        name="file_system",
        action="write",
        input={
            "file_path": assemble_project_path(f"workdir/{repo_name}/test.txt"),
            "content": "Hello from ECP workflow test!\nThis is a test file created by the workflow."
        }
    )
    print(f"Result: {res3}")
    
    # Step 4: Add, commit and push changes to remote
    print("\n💾 Step 4: Add, commit and push changes to remote")
    res4 = await ecp.ainvoke(
        name="github",
        action="git_commit",
        input={
            "local_path": assemble_project_path(f"workdir/{repo_name}"),
            "message": "Add test file from ECP workflow",
            "add_all": True
        }
    )
    print(f"Commit Result: {res4}")
    
    # Push changes to remote repository
    res5 = await ecp.ainvoke(
        name="github",
        action="git_push",
        input={
            "local_path": assemble_project_path(f"workdir/{repo_name}"),
            "remote": "origin",
            "branch": "main"
        }
    )
    print(f"Push Result: {res5}")
    
    print("\n✅ GitHub Environment Workflow Test Completed!")
    print("=" * 50)
    
async def test_operator_browser():
    
    
    state = await ecp.get_state("operator_browser")
    logger.info(f"| 📝 State: {state}")
        
    res = await ecp.ainvoke(
        name="operator_browser",
        action="type",
        input={
            "text": "python programming"
        }
    )
    logger.info(f"| 📝 Result: {res}")
    
    
async def main():
    
    args = parse_args()
    
    # Initialize configuration
    config.init_config(args.config, args)
    logger.init_logger(config)
    logger.info(f"| Config: {config.pretty_text}")
    
    # Initialize model manager
    logger.info("| 🧠 Initializing model manager...")
    await model_manager.initialize(use_local_proxy=config.use_local_proxy)
    logger.info(f"| ✅ Model manager initialized: {model_manager.list()}")
    
    # Initialize tool manager
    logger.info("| 🛠️ Initializing tool manager...")
    await tcp.initialize(config.tool_names)
    logger.info(f"| ✅ Tool manager initialized: {tcp.list()}")
    
    # Initialize environments
    logger.info("| 🎮 Initializing environments...")
    await ecp.initialize(config.env_names)
    logger.info(f"| ✅ Environments initialized: {ecp.list()}")
    
    # Test file system
    # await test_file_system()
    # await test_github()
    await test_operator_browser()
    

if __name__ == "__main__":
    asyncio.run(main())
