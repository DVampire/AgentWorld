"""Example script demonstrating how to use the LeetCode tool to fetch problem information."""

import os
import sys
from dotenv import load_dotenv
load_dotenv(verbose=True)

from pathlib import Path
import asyncio

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.tool.default_tools.leetcode import LeetCodeTool


async def main():
    """Main function to demonstrate LeetCode tool usage."""
    # Initialize the LeetCode tool
    leetcode_tool = LeetCodeTool()
    
    print("=" * 80)
    print("LeetCode Tool Example")
    print("=" * 80)
    print()
    
    # Example 1: Fetch problem by slug
    print("Example 1: Fetching problem by slug ('two-sum')")
    print("-" * 80)
    response1 = await leetcode_tool(slug="two-sum")
    if response1.success:
        print(response1.message)
        print()
    else:
        print(f"Error: {response1.message}")
        print()
    
    # Example 2: Fetch problem by ID
    print("Example 2: Fetching problem by ID (1)")
    print("-" * 80)
    response2 = await leetcode_tool(problem_id=1)
    if response2.success:
        print(response2.message)
        print()
    else:
        print(f"Error: {response2.message}")
        print()
    
    # Example 3: Fetch another problem
    print("Example 3: Fetching problem by slug ('reverse-linked-list')")
    print("-" * 80)
    response3 = await leetcode_tool(slug="reverse-linked-list")
    if response3.success:
        print(response3.message)
        print()
    else:
        print(f"Error: {response3.message}")
        print()


if __name__ == "__main__":
    asyncio.run(main())

