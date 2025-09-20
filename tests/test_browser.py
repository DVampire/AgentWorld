"""Test script for browser tool functionality."""

import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(verbose=True)

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.environments.playwright.service import PlaywrightService

async def main():
	# 使用 ScreenshotService 来管理截屏
	service = PlaywrightService(
		screenshots_dir="workdir/screenshots"
	)
	
	try:
		print("🚀 Starting browser...")
		await service.start()
		print("✅ Browser started successfully")
		
		print("🌐 Navigating to Google...")
		result = await service.go_to_url('https://www.google.com')
		print(f"📄 Navigation result: {result}")
		
		print("📸 Taking screenshot with element highlights using ScreenshotService...")
		highlighted_path = await service.store_screenshot(1, highlight_elements=True)
		print(f"🖼️ Highlighted screenshot saved to: {highlighted_path}")
		
		print("🔍 Testing search functionality...")
		# 使用 search_google 方法进行搜索
		search_result = await service.search_google("Python programming")
		print(f"🔍 Search result: {search_result}")
  
		await asyncio.sleep(2)
		
		print("📸 Taking screenshot after search using ScreenshotService...")
		search_screenshot = await service.store_screenshot(2, highlight_elements=True)
		print(f"🖼️ Search result screenshot saved to: {search_screenshot}")
		
		print("🔍 Getting browser state to see available elements...")
		browser_state = await service.state(include_screenshot=True)
		print(browser_state)
		
		print("✅ Test completed successfully!")
		
	except Exception as e:
		print(f"❌ Error occurred: {e}")
		import traceback
		traceback.print_exc()
	finally:
		print("🔒 Closing browser...")
		await service.close()
		print("✅ Browser closed")

if __name__ == '__main__':
	asyncio.run(main())