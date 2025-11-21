"""Test script for the report tool.

This script demonstrates three ways to test the report tool:
1. Direct standalone execution (using main function)
2. Direct tool invocation (using tcp.ainvoke)
3. Through agent (using agent with report tool)
"""

import os
import sys
from dotenv import load_dotenv
load_dotenv(verbose=True)

from pathlib import Path
import asyncio

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.config import config
from src.logger import logger
from src.tools import tcp
from src.models import model_manager


async def test_direct_tool_invocation():
    """Test the report tool directly using tcp.ainvoke"""
    print("\n" + "="*60)
    print("🧪 Test 1: Direct Tool Invocation")
    print("="*60)
    
    # Initialize model manager (required for some tools)
    logger.info("| 🧠 Initializing model manager...")
    await model_manager.initialize(use_local_proxy=False)
    
    # Initialize tools
    logger.info("| 🛠️ Initializing tools...")
    await tcp.initialize(["report"])
    logger.info(f"| ✅ Tools initialized: {tcp.list()}")
    
    # Test query
    query = "What is the latest news about Apple stock price and market analysis?"
    
    try:
        # Invoke the report tool directly
        result = await tcp.ainvoke("report", {"query": query})
        
        print(f"\n📋 Query: {query}")
        print("\n📄 Result:")
        print("-" * 60)
        if hasattr(result, 'message'):
            print(result.message)
        else:
            print(result)
        print("-" * 60)
        
        if hasattr(result, 'extra') and result.extra:
            print("\n📁 File Information:")
            if 'path' in result.extra:
                print(f"  Path: {result.extra['path']}")
            if 'absolute_path' in result.extra:
                print(f"  Absolute Path: {result.extra['absolute_path']}")
            if 'html_length' in result.extra:
                print(f"  HTML Length: {result.extra['html_length']} characters")
        
        print("\n✅ Direct tool invocation test completed!")
        
    except Exception as e:
        print(f"\n❌ Error testing report tool: {e}")
        import traceback
        traceback.print_exc()


async def test_with_custom_output_path():
    """Test the report tool with custom output path"""
    print("\n" + "="*60)
    print("🧪 Test 2: Custom Output Path")
    print("="*60)
    
    # Initialize tools
    await tcp.initialize(["report"])
    
    query = "Tesla stock analysis and recent market trends"
    custom_path = "workdir/test_reports/custom_report.html"
    
    try:
        result = await tcp.ainvoke("report", {
            "query": query,
            "output_path": custom_path
        })
        
        print(f"\n📋 Query: {query}")
        print(f"📁 Custom Path: {custom_path}")
        print("\n📄 Result:")
        print("-" * 60)
        print(result.message if hasattr(result, 'message') else result)
        print("-" * 60)
        
        print("\n✅ Custom output path test completed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_standalone_execution():
    """Test standalone execution (using main function)"""
    print("\n" + "="*60)
    print("🧪 Test 3: Standalone Execution")
    print("="*60)
    print("\nTo test standalone execution, run:")
    print("  python src/tools/report/report.py \"Your query here\"")
    print("\nExample:")
    print("  python src/tools/report/report.py \"What is the latest news about Apple?\"")
    print("\nNote: This requires environment variables to be set in .env file")


async def main():
    """Main test function"""
    print("\n" + "="*60)
    print("📊 Report Tool Test Suite")
    print("="*60)
    
    # Check environment variables
    print("\n🔍 Checking environment variables...")
    required_vars = ["OPENAI_API_KEY", "TAVILY_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("Please set them in your .env file or environment")
        print("\nRequired variables:")
        print("  - OPENAI_API_KEY (or REPORT_ENGINE_API_KEY)")
        print("  - TAVILY_API_KEY")
        print("\nOptional variables:")
        print("  - OPENAI_BASE_URL (or REPORT_ENGINE_BASE_URL)")
        print("  - OPENAI_MODEL_NAME (or REPORT_ENGINE_MODEL_NAME)")
    else:
        print("✅ All required environment variables are set")
    
    # Run tests
    try:
        # Test 1: Direct tool invocation
        await test_direct_tool_invocation()
        
        # Test 2: Custom output path
        await test_with_custom_output_path()
        
        # Test 3: Standalone execution info
        test_standalone_execution()
        
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("✅ Test suite completed!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())

