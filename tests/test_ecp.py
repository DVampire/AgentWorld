"""
Demonstration of the Environment Context Protocol (ECP) usage.
"""
from pathlib import Path
import sys
import asyncio

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.environments import ecp
from src.utils import assemble_project_path
    
async def main():
    ecp.build_environment("file_system", env_config={"base_dir": assemble_project_path("tests/files")})
    print(ecp.list_actions())
    
    input = {
        "env_name": "file_system",
        "action_name": "read",
        "file_path": assemble_project_path("tests/files/test.txt")
    }
    
    result = await ecp.call_action(**input)
    print(result)
    
if __name__ == "__main__":
    asyncio.run(main())