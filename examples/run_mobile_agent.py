"""Example script to run mobile agent."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.environments.mobile import MobileService, TapRequest, SwipeRequest, InputTextRequest


async def main():
    """Main function to test mobile service."""
    print("🚀 Starting Mobile Agent...")
    print("📋 Prerequisites:")
    print("   - Android device connected via USB")
    print("   - USB debugging enabled")
    print("   - adbutils package installed: pip install adbutils")
    print("   - ADB tools installed and in PATH")
    print()
    
    # Initialize mobile service
    mobile_service = MobileService(
        device_id=None,  # Use first connected device
        video_save_path="./workdir/mobile_agent/videos",
        video_save_name="test_record",
        fps=30,
        base_dir="./workdir/mobile_agent"
    )
    
    try:
        # Start the service
        print("📱 Connecting to device...")
        success = await mobile_service.start()
        
        if not success:
            print("❌ Failed to connect to device")
            print("💡 Make sure:")
            print("   - Device is connected via USB")
            print("   - USB debugging is enabled")
            print("   - adbutils is installed: pip install adbutils")
            return
        
        print("✅ Device connected successfully!")
        
        # Get device info
        device_state = await mobile_service.get_device_state()
        print(f"📱 Device: {device_state.device_info.device_id}")
        print(f"📏 Screen: {device_state.device_info.screen_width}x{device_state.device_info.screen_height}")
        print(f"🔍 Density: {device_state.device_info.screen_density}")
        
        # Test tap action
        print("\n👆 Testing tap action...")
        tap_result = await mobile_service.tap(TapRequest(x=500, y=500))
        print(f"Tap result: {tap_result.success} - {tap_result.message}")
        if tap_result.screenshot_path:
            print(f"📸 Screenshot: {tap_result.screenshot_path}")
        
        # Test swipe action
        print("\n👆 Testing swipe action...")
        swipe_result = await mobile_service.swipe(SwipeRequest(
            start_x=100, start_y=500,
            end_x=900, end_y=500,
            duration=500
        ))
        print(f"Swipe result: {swipe_result.success} - {swipe_result.message}")
        if swipe_result.screenshot_path:
            print(f"📸 Screenshot: {swipe_result.screenshot_path}")
        
        # Test text input
        print("\n⌨️ Testing text input...")
        text_result = await mobile_service.input_text(InputTextRequest(text="Hello Mobile!"))
        print(f"Text input result: {text_result.success} - {text_result.message}")
        if text_result.screenshot_path:
            print(f"📸 Screenshot: {text_result.screenshot_path}")
        
        # Take a screenshot
        print("\n📸 Taking screenshot...")
        from src.environments.mobile.types import ScreenshotRequest
        screenshot_result = await mobile_service.take_screenshot(ScreenshotRequest())
        print(f"Screenshot result: {screenshot_result.success} - {screenshot_result.message}")
        if screenshot_result.screenshot_path:
            print(f"📸 Screenshot saved: {screenshot_result.screenshot_path}")
        
        print("\n✅ Mobile agent test completed successfully!")
        print("🎥 Video recording saved to: ./workdir/mobile_agent/videos/")
        
    except Exception as e:
        print(f"❌ Error during mobile agent test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Stop the service
        print("\n🛑 Stopping mobile service...")
        await mobile_service.stop()
        print("✅ Mobile service stopped")


if __name__ == "__main__":
    asyncio.run(main())
