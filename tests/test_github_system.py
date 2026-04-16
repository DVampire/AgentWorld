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
from src.models import model_manager
from src.utils import get_env
from src.environments import GitHubEnvironment

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

async def test_github_system():
    # secret_from_env returns a function, need to call it to get the actual value
    token = get_env("GITHUB_TOKEN")
    username = get_env("GITHUB_USERNAME")
    
    print(f"| Token: {token}")
    print(f"| Username: {username}")
    
    try:
        github_env = GitHubEnvironment(
            token=token.get_secret_value(),
            username=username.get_secret_value()
        )
        await github_env.initialize()
        logger.info(f"| GitHub Environment: {github_env}")
        
        # Test some basic functionality
        logger.info("| Testing GitHub environment...")
        await github_env._create_repository(
            name="test-repo",
            description="Test repository",
            private=True
        )
        
    except ValueError as e:
        print(f"| Error getting secrets: {e}")
        print("| Please set GITHUB_TOKEN and GITHUB_USERNAME environment variables")
        return
    except Exception as e:
        logger.error(f"| Error initializing GitHub environment: {e}")
        return
    finally:
        # Ensure proper cleanup
        if 'github_env' in locals():
            await github_env.cleanup()
            logger.info("| GitHub environment cleaned up")

async def main():
    args = parse_args()
    
    config.init_config(args.config, args)
    logger.init_logger(config)
    logger.info(f"| Config: {config.pretty_text}")
    
    await model_manager.init_models(use_local_proxy=config.use_local_proxy)
    logger.info(f"| Models: {model_manager.list_models()}")
    
    await test_github_system()
    
if __name__ == "__main__":
    asyncio.run(main())