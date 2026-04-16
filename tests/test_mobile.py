import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(verbose=True)

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.environments.mobile import (
    MobileService, 
    TapRequest, SwipeRequest, PressRequest, 
    TypeRequest, KeyEventRequest, ScreenshotRequest
)


async def test_mobile_operations():
    """Test all mobile operations."""
    print("🧪 Mobile Operations Test Suite")
    print("=" * 50)
    
    # Initialize mobile service
    mobile_service = MobileService(
        base_dir="./workdir/mobile_test",
        device_id=None,  # Use first connected device
        fps=2,
        bitrate=50000000,
        chunk_duration=60
    )
    
    try:
        # Start service
        print("🚀 Starting mobile service...")
        success = await mobile_service.start()
        if not success:
            print("❌ Failed to start mobile service")
            return False
        
        print("✅ Mobile service started successfully")
        
        # Get device state
        print("\n📱 Getting device state...")
        state = await mobile_service.get_device_state()
        print(f"   Device: {state.device_info.device_id}")
        print(f"   Screen: {state.device_info.screen_width}x{state.device_info.screen_height}")
        print(f"   Density: {state.device_info.screen_density}")
        
        # Test basic operations
        print("\n👆 Testing Basic Operations:")
        
        # Test tap
        print("   - Testing tap...")
        tap_result = await mobile_service.tap(TapRequest(x=500, y=500))
        print(f"     Result: {tap_result.success} - {tap_result.message}")
        await asyncio.sleep(1)
        
        # Test swipe
        print("   - Testing swipe...")
        swipe_result = await mobile_service.swipe(SwipeRequest(
            start_x=100, start_y=500,
            end_x=900, end_y=500,
            duration=500
        ))
        print(f"     Result: {swipe_result.success} - {swipe_result.message}")
        await asyncio.sleep(1)
        
        # Test press (long press)
        print("   - Testing press (long press)...")
        press_result = await mobile_service.press(PressRequest(
            x=500, y=500, duration=1000
        ))
        print(f"     Result: {press_result.success} - {press_result.message}")
        await asyncio.sleep(1)
        
        # Test type text
        print("   - Testing type text...")
        type_result = await mobile_service.type(TypeRequest(text="Hello Mobile World!"))
        print(f"     Result: {type_result.success} - {type_result.message}")
        await asyncio.sleep(1)
        
        # Test key event
        print("   - Testing key event...")
        key_result = await mobile_service.key_event(KeyEventRequest(keycode=4))  # KEYCODE_BACK
        print(f"     Result: {key_result.success} - {key_result.message}")
        await asyncio.sleep(1)
        
        # Test screenshot
        print("   - Testing screenshot...")
        screenshot_result = await mobile_service.take_screenshot(ScreenshotRequest())
        print(f"     Result: {screenshot_result.success} - {screenshot_result.message}")
        if screenshot_result.success:
            print(f"     Screenshot saved: {screenshot_result.screenshot_path}")
        
        # Test scroll operations
        print("\n📜 Testing Scroll Operations:")
        
        # Test scroll up
        print("   - Testing scroll up...")
        await mobile_service.adb.scroll_up()
        await asyncio.sleep(0.5)
        
        # Test scroll down
        print("   - Testing scroll down...")
        await mobile_service.adb.scroll_down()
        await asyncio.sleep(0.5)
        
        # Test scroll left
        print("   - Testing scroll left...")
        await mobile_service.adb.scroll_left()
        await asyncio.sleep(0.5)
        
        # Test scroll right
        print("   - Testing scroll right...")
        await mobile_service.adb.scroll_right()
        await asyncio.sleep(0.5)
        
        # Test system operations
        print("\n🔧 Testing System Operations:")
        
        # Test wake up
        print("   - Testing wake up...")
        await mobile_service.adb.wake_up()
        await asyncio.sleep(1)
        
        # Test unlock screen
        print("   - Testing unlock screen...")
        await mobile_service.adb.unlock_screen()
        await asyncio.sleep(1)
        
        # Test device info
        print("\n📊 Testing Device Info:")
        device_info = await mobile_service.adb.get_device_info()
        print(f"   Model: {device_info.get('model', 'Unknown')}")
        print(f"   Version: {device_info.get('version', 'Unknown')}")
        print(f"   Screen Size: {device_info.get('screen_size', 'Unknown')}")
        print(f"   Screen Density: {device_info.get('screen_density', 'Unknown')}")
        
        current_activity = await mobile_service.adb.get_current_activity()
        print(f"   Current Activity: {current_activity}")
        
        print("\n✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Stop service
        print("\n🛑 Stopping mobile service...")
        await mobile_service.stop()
        print("✅ Mobile service stopped")


async def main():
    """Main test function."""
    print("📋 Prerequisites:")
    print("   - Android device connected via USB")
    print("   - USB debugging enabled")
    print("   - adbutils package installed")
    print("   - scrcpy installed (optional, for advanced features)")
    print()
    
    success = await test_mobile_operations()
    
    if success:
        print("\n🎉 All mobile operations tested successfully!")
    else:
        print("\n💥 Some tests failed. Check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())