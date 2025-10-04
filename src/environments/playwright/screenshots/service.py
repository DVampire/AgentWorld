"""
Screenshot storage service for browser-use agents.
"""

import base64
from pathlib import Path
from typing import Union
from PIL import Image, ImageDraw, ImageFilter
import anyio
import io
import cairosvg

from src.environments.playwright.observability import observe_debug

class ScreenshotService:
	"""Simple screenshot storage service that saves screenshots to disk"""

	def __init__(self, agent_directory: Union[str, Path]):
		"""Initialize with agent directory path"""
		self.agent_directory = Path(agent_directory) if isinstance(agent_directory, str) else agent_directory

		# Create screenshots subdirectory
		self.screenshots_dir = self.agent_directory / 'screenshots'
		self.screenshots_dir.mkdir(parents=True, exist_ok=True)

	@observe_debug(ignore_input=True, ignore_output=True, name='store_screenshot')
	async def store_screenshot(self, screenshot_b64: str, step_number: int) -> str:
		"""Store screenshot to disk and return the full path as string"""
		screenshot_filename = f'step_{step_number:04d}.png'
		screenshot_path = self.screenshots_dir / screenshot_filename

		# Decode base64 and save to disk
		screenshot_data = base64.b64decode(screenshot_b64)

		async with await anyio.open_file(screenshot_path, 'wb') as f:
			await f.write(screenshot_data)

		return str(screenshot_path)

	@observe_debug(ignore_input=True, ignore_output=True, name='get_screenshot_from_disk')
	async def get_screenshot(self, screenshot_path: str) -> str | None:
		"""Load screenshot from disk path and return as base64"""
		if not screenshot_path:
			return None

		path = Path(screenshot_path)
		if not path.exists():
			return None

		# Load from disk and encode to base64
		async with await anyio.open_file(path, 'rb') as f:
			screenshot_data = await f.read()

		return base64.b64encode(screenshot_data).decode('utf-8')

	async def draw_cursor(self, screenshot_path: str, x: int, y: int, size: int = 32) -> str:
		"""
		Draw a Mac-style cursor on the screenshot using SVG.
		
		Args:
			screenshot_path: Path to the screenshot
			x: X coordinate of the cursor
			y: Y coordinate of the cursor
			size: Size of the cursor
			
		Returns:
			Path to the processed screenshot
		"""
		
		# Load the screenshot
		base_img = Image.open(screenshot_path)
		
		# Create SVG cursor (already at 120 degree angle)
		svg_code = f'''
		<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 32" width="{size}" height="{size}">
			<defs>
				<filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
					<feDropShadow dx="2" dy="2" stdDeviation="1" flood-color="rgba(0,0,0,0.3)"/>
				</filter>
			</defs>
			<path d="M0,0 L0,26 L6,20 L11,32 L14,31 L9,19 L18,19 Z"
				  fill="black" stroke="white" stroke-width="1.2" filter="url(#shadow)"/>
		</svg>
		'''
		
		# Convert SVG to PNG
		svg_bytes = svg_code.encode('utf-8')
		cursor_png = cairosvg.svg2png(bytestring=svg_bytes)
		cursor_img = Image.open(io.BytesIO(cursor_png))
		
		# Calculate position to center the cursor at the target coordinates
		cursor_x = x - size // 2
		cursor_y = y - size // 2
		
		# Ensure cursor stays within image bounds
		cursor_x = max(0, min(cursor_x, base_img.width - size))
		cursor_y = max(0, min(cursor_y, base_img.height - size))
		
		# Paste cursor onto screenshot
		if cursor_img.mode == 'RGBA':
			base_img.paste(cursor_img, (cursor_x, cursor_y), cursor_img)
		else:
			base_img.paste(cursor_img, (cursor_x, cursor_y))
		
		# Save the processed screenshot
		base_img.save(screenshot_path)
		
		return screenshot_path

	async def draw_path(self, screenshot_path: str, path: list[list[int]], arrow_size: int = 16) -> str:
		"""
		Draw a path on the screenshot with arrows showing direction.
		
		Args:
			screenshot_path: Path to the screenshot
			path: List of [x, y] coordinates representing the path
			arrow_size: Size of the direction arrows
			
		Returns:
			Path to the processed screenshot
		"""
		import math
		
		# Load the screenshot
		base_img = Image.open(screenshot_path)
		draw = ImageDraw.Draw(base_img)
		
		if len(path) < 2:
			return screenshot_path
		
		# Draw the path line
		for i in range(len(path) - 1):
			start = path[i]
			end = path[i + 1]
			# Draw line segment
			draw.line([start[0], start[1], end[0], end[1]], fill=(255, 0, 0, 255), width=3)
		
		# Draw direction arrows along the path
		for i in range(len(path) - 1):
			start = path[i]
			end = path[i + 1]
			
			# Calculate direction vector
			dx = end[0] - start[0]
			dy = end[1] - start[1]
			distance = math.sqrt(dx*dx + dy*dy)
			
			if distance == 0:
				continue
				
			# Normalize direction vector
			dx_norm = dx / distance
			dy_norm = dy / distance
			
			# Calculate angle for arrow
			angle = math.degrees(math.atan2(dy_norm, dx_norm))
			
			# Position arrow at midpoint of segment
			mid_x = (start[0] + end[0]) // 2
			mid_y = (start[1] + end[1]) // 2
			
			# Draw arrow at midpoint
			await self._draw_arrow_at_position(draw, mid_x, mid_y, angle, arrow_size)
		
		# Draw numbered points
		for i, point in enumerate(path):
			x, y = point
			# Draw circle for point
			radius = 8
			draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=(0, 255, 0, 255), outline=(0, 0, 0, 255), width=2)
			
			# Draw number
			text = str(i + 1)
			# Simple text drawing (you might want to use a proper font)
			text_bbox = draw.textbbox((0, 0), text)
			text_width = text_bbox[2] - text_bbox[0]
			text_height = text_bbox[3] - text_bbox[1]
			text_x = x - text_width // 2
			text_y = y - text_height // 2
			draw.text((text_x, text_y), text, fill=(255, 255, 255, 255))
		
		# Save the processed screenshot
		base_img.save(screenshot_path)
		
		return screenshot_path
	
	async def _draw_arrow_at_position(self, draw, x: int, y: int, angle: float, size: int):
		"""Draw a small arrow at the specified position and angle."""
		import math
		
		# Create arrow shape relative to origin
		arrow_points = [
			(0, 0),  # tip
			(-size//2, -size//4),  # left edge
			(-size//4, -size//2),  # left back
			(size//4, -size//2),   # right back
			(size//2, -size//4),   # right edge
		]
		
		# Rotate points around origin
		angle_rad = math.radians(angle)
		cos_a = math.cos(angle_rad)
		sin_a = math.sin(angle_rad)
		
		rotated_points = []
		for px, py in arrow_points:
			# Rotate around origin
			rx = px * cos_a - py * sin_a
			ry = px * sin_a + py * cos_a
			# Translate to target position
			rotated_points.append((int(x + rx), int(y + ry)))
		
		# Draw arrow
		draw.polygon(rotated_points, fill=(255, 0, 0, 255), outline=(0, 0, 0, 255), width=1)

	async def draw_scroll(self, screenshot_path: str, x: int, y: int, scroll_x: int, scroll_y: int) -> str:
		"""
		Draw a scroll operation on the screenshot.
		
		Args:
			screenshot_path: Path to the screenshot
			x: X coordinate where scroll started
			y: Y coordinate where scroll started
			scroll_x: X scroll delta (positive = right, negative = left)
			scroll_y: Y scroll delta (positive = down, negative = up)
			
		Returns:
			Path to the processed screenshot
		"""
		import math
		
		# Load the screenshot
		base_img = Image.open(screenshot_path)
		draw = ImageDraw.Draw(base_img)
		
		# Calculate scroll direction and magnitude
		scroll_magnitude = math.sqrt(scroll_x * scroll_x + scroll_y * scroll_y)
		if scroll_magnitude == 0:
			return screenshot_path
		
		# Normalize scroll direction
		scroll_dx = scroll_x / scroll_magnitude
		scroll_dy = scroll_y / scroll_magnitude
		
		# Calculate scroll angle
		scroll_angle = math.degrees(math.atan2(scroll_dy, scroll_dx))
		
		# Draw scroll indicator circle
		circle_radius = 20
		draw.ellipse([x-circle_radius, y-circle_radius, x+circle_radius, y+circle_radius], 
					fill=(0, 0, 255, 100), outline=(0, 0, 255, 255), width=3)
		
		# Draw scroll direction arrow
		arrow_size = 30
		await self._draw_arrow_at_position(draw, x, y, scroll_angle, arrow_size)
		
		# Draw scroll magnitude indicator (line length proportional to scroll amount)
		max_scroll_indicator = 100  # Maximum length for indicator line
		indicator_length = min(scroll_magnitude, max_scroll_indicator)
		
		# Calculate end point of indicator line
		indicator_end_x = int(x + scroll_dx * indicator_length)
		indicator_end_y = int(y + scroll_dy * indicator_length)
		
		# Draw indicator line
		draw.line([x, y, indicator_end_x, indicator_end_y], fill=(0, 0, 255, 255), width=4)
		
		# Draw scroll text
		scroll_text = f"Scroll: {scroll_x:+d}, {scroll_y:+d}"
		text_bbox = draw.textbbox((0, 0), scroll_text)
		text_width = text_bbox[2] - text_bbox[0]
		text_height = text_bbox[3] - text_bbox[1]
		
		# Position text near the scroll point
		text_x = x + circle_radius + 10
		text_y = y - text_height // 2
		
		# Draw text background
		draw.rectangle([text_x-5, text_y-2, text_x+text_width+5, text_y+text_height+2], 
					  fill=(255, 255, 255, 200), outline=(0, 0, 0, 255), width=1)
		draw.text((text_x, text_y), scroll_text, fill=(0, 0, 0, 255))
		
		# Save the processed screenshot
		base_img.save(screenshot_path)
		
		return screenshot_path