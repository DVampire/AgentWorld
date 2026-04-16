import os
import sys
from dotenv import load_dotenv
load_dotenv(verbose=True)

from pathlib import Path
import asyncio
import signal
import time

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.environments.operator_browser.service import OperatorBrowserService

async def test_browser_startup():
    print("🔍 Starting browser startup test...")
    
    browser = OperatorBrowserService(
        base_dir=os.path.join(root, "workdir", "operator_browser"),
        headless=False,
        viewport={"width": 1280, "height": 720},
    )
    
    try:
        print("📱 Calling browser.start()...")
        start_time = time.time()
        
        # Add timeout to prevent infinite hang
        await asyncio.wait_for(browser.start(), timeout=30.0)
        
        end_time = time.time()
        print(f"✅ Browser started successfully in {end_time - start_time:.2f} seconds")
        
        await asyncio.sleep(5)
        
        # Try to get browser state
        print("📊 Getting browser state...")
        state = await browser.get_state()
        print(f"✅ Browser state: Screenshot={'Yes' if state['screenshot'] else 'No'}")
        print(f"Screenshot length: {len(state['screenshot'])}")
        
        await browser.stop()
        
    except asyncio.TimeoutError:
        print("❌ Browser startup timed out after 30 seconds")
    except Exception as e:
        print(f"❌ Error during browser startup: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🔄 Attempting to kill browser...")
        try:
            await browser.stop()
            print("✅ Browser killed successfully")
        except Exception as e:
            print(f"⚠️ Error killing browser: {e}")

if __name__ == "__main__":
    # Set up signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print("\n🛑 Received interrupt signal, exiting...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    asyncio.run(test_browser_startup())
