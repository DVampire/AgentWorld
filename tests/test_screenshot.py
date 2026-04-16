import os
import sys
from dotenv import load_dotenv
load_dotenv(verbose=True)

from pathlib import Path
import asyncio
import time

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.environments.browser import Browser

async def test_screenshot():
    print("🔍 Starting screenshot test...")
    
    browser = Browser()
    
    try:
        print("📱 Starting browser...")
        await browser.start()
        print("✅ Browser started")
        
        page = await browser.get_current_page()
        print("✅ Got current page")
        
        # Navigate to a real page
        print("🌐 Navigating to Google...")
        await page.goto("https://www.google.com")
        await asyncio.sleep(2)  # Wait for page to load
        print("✅ Navigation complete")
        
        # Test 1: get_browser_state_summary with screenshot
        print("📸 Testing get_browser_state_summary...")
        try:
            state = await asyncio.wait_for(
                browser.get_browser_state_summary(include_screenshot=True),
                timeout=10.0
            )
            print(f"✅ State summary: URL={state.url}, Screenshot={'Yes' if state.screenshot else 'No'}")
            if state.screenshot:
                print(f"📏 Screenshot length: {len(state.screenshot)}")
            else:
                print("❌ Screenshot is None")
        except asyncio.TimeoutError:
            print("❌ get_browser_state_summary timed out")
        except Exception as e:
            print(f"❌ get_browser_state_summary failed: {e}")
        
        # Test 2: Direct screenshot capture
        print("📸 Testing direct screenshot capture...")
        try:
            if hasattr(browser, '_dom_watchdog') and browser._dom_watchdog:
                print("🔍 DOMWatchdog found, trying direct capture...")
                direct_screenshot = await asyncio.wait_for(
                    browser._dom_watchdog._capture_clean_screenshot(),
                    timeout=10.0
                )
                if direct_screenshot:
                    print(f"✅ Direct screenshot captured, length: {len(direct_screenshot)}")
                else:
                    print("❌ Direct screenshot is None")
            else:
                print("❌ No DOMWatchdog found")
        except asyncio.TimeoutError:
            print("❌ Direct screenshot capture timed out")
        except Exception as e:
            print(f"❌ Direct screenshot capture failed: {e}")
        
        # Test 3: Check CDP connection
        print("🔗 Testing CDP connection...")
        try:
            if hasattr(browser, '_cdp_client_root') and browser._cdp_client_root:
                targets = await asyncio.wait_for(
                    browser._cdp_client_root.send.Target.getTargets(),
                    timeout=5.0
                )
                print(f"✅ CDP connection active, {len(targets.get('targetInfos', []))} targets")
            else:
                print("❌ No CDP client root found")
        except Exception as e:
            print(f"❌ CDP connection test failed: {e}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🔄 Killing browser...")
        try:
            await browser.kill()
            print("✅ Browser killed")
        except Exception as e:
            print(f"⚠️ Error killing browser: {e}")

if __name__ == "__main__":
    asyncio.run(test_screenshot())
