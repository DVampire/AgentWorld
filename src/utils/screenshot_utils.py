"""
Screenshot storage service for browser-use agents.
"""

from pathlib import Path
from typing import Union, Optional, Dict, Any, Tuple
from PIL import Image, ImageDraw
import io
import cairosvg

class ScreenshotService:
	"""Simple screenshot storage service that saves screenshots to disk"""

	def __init__(self, 
              base_dir: Union[str, Path],
              adapt_window_size: bool = False,
              target_window_width: Optional[int] = None,
              target_window_height: Optional[int] = None,
              pad_color: Optional[tuple[int, int, int]] = None):
		"""Initialize with agent directory path"""
		self.base_dir = Path(base_dir) if isinstance(base_dir, str) else base_dir

		# Create screenshots subdirectory
		self.screenshots_dir = self.base_dir / 'screenshots'
		self.screenshots_dir.mkdir(parents=True, exist_ok=True)
	
		# Adapt window size
		self.adapt_window_size = adapt_window_size
		self.target_window_width = target_window_width or 1920
		self.target_window_height = target_window_height or 1080
		self.pad_color = pad_color or (0, 0, 0)
  
	def compute_transform_info(self, 
                            source_width: int,
                            source_height: int,
                            target_width: int = 1920,
                            target_height: int = 1080) -> Dict[str, Any]:
		"""Compute the transform information for the screenshot."""
		target_ratio = target_width / target_height
		src_ratio = source_width / source_height

		if src_ratio > target_ratio: # The source image is wider than the target image
			new_width = target_width
			new_height = int(target_width / src_ratio)
			scale = new_width / source_width
			paste_x = 0
			paste_y = (target_height - new_height) // 2
		else:  # The source image is higher or narrower than the target image
			new_height = target_height
			new_width = int(target_height * src_ratio)
			scale = new_height / source_height
			paste_x = (target_width - new_width) // 2
			paste_y = 0

		return {
			"scale": scale,
			"paste_x": paste_x,
			"paste_y": paste_y,
		}
  
	def transform_screenshot(self, 
	                        img: Image.Image,
	                        target_width: int = 1920, 
	                        target_height: int = 1080, 
	                        pad_color: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
		"""
		Resize and pad an image to a fixed target width and height.
		
		Args:
			img: PIL Image
			target_width: Target width
			target_height: Target height
			pad_color: Fill color (R, G, B)
   
		Returns:
			Base64 encoded string of the resized and padded screenshot
		"""
		
		source_width, source_height = img.size

		transform_info = self.compute_transform_info(source_width, source_height, target_width, target_height)
		scale = transform_info["scale"]
		paste_x = transform_info["paste_x"]
		paste_y = transform_info["paste_y"]

		new_width = int(source_width * scale)
		new_height = int(source_height * scale)

		resized = img.resize((new_width, new_height), Image.LANCZOS)
		new_img = Image.new("RGB", (target_width, target_height), pad_color)
		new_img.paste(resized, (paste_x, paste_y))
  
		return new_img

	def transform_point(self, 
	                   x: int,
	                   y: int,
	                   source_width: int,
	                   source_height: int,
	                   target_width: int = 1920, 
	                   target_height: int = 1080) -> Tuple[int, int]:
		"""
		Transform a point from source coordinates to target coordinates.
		
		Args:
			x: Source X coordinate
			y: Source Y coordinate
			source_width: Source image width
			source_height: Source image height
			target_width: Target image width
			target_height: Target image height
			
		Returns:
			Tuple of (target_x, target_y) coordinates
		"""
		# Clamp input coordinates to source bounds
		x = max(0, min(x, source_width - 1))
		y = max(0, min(y, source_height - 1))
		
		transform_info = self.compute_transform_info(source_width, source_height, target_width, target_height)
		scale = transform_info["scale"]
		paste_x = transform_info["paste_x"]
		paste_y = transform_info["paste_y"]

		new_x = int(x * scale) + paste_x
		new_y = int(y * scale) + paste_y
		
		# Ensure coordinates are within target bounds
		new_x = max(0, min(new_x, target_width - 1))
		new_y = max(0, min(new_y, target_height - 1))

		return new_x, new_y

	def inverse_transform_point(self, 
	                                 x: int,
	                                 y: int,
	                                 source_width: int,
	                                 source_height: int,
	                                 target_width: int = 1920, 
	                                 target_height: int = 1080) -> Tuple[int, int]:
		"""
		Inverse transform a point to a new position based on the transform information.
		"""
		transform_info = self.compute_transform_info(source_width, source_height, target_width, target_height)
		scale = transform_info["scale"]
		paste_x = transform_info["paste_x"]
		paste_y = transform_info["paste_y"]

		# Calculate the actual image area in target coordinates
		actual_width = int(source_width * scale)
		actual_height = int(source_height * scale)
		
		# Clamp coordinates to the actual image area first
		clamped_x = max(paste_x, min(x, paste_x + actual_width - 1))
		clamped_y = max(paste_y, min(y, paste_y + actual_height - 1))
		
		# Transform to source coordinates
		new_x = int((clamped_x - paste_x) / scale)
		new_y = int((clamped_y - paste_y) / scale)
		
		# Ensure coordinates are within source bounds
		new_x = max(0, min(new_x, source_width - 1))
		new_y = max(0, min(new_y, source_height - 1))

		return new_x, new_y

	async def store_screenshot(self, 
                            img: Image.Image, 
                            step_number: Optional[int] = None, 
                            screenshot_filename: Optional[str] = None) -> str:
		"""Store screenshot to disk and return the full path as string"""
		if screenshot_filename:
			screenshot_filename = f'{screenshot_filename}'
		else:
			screenshot_filename = f'step_{step_number:04d}.png'
		screenshot_path = self.screenshots_dir / screenshot_filename
  
		img.save(screenshot_path)

		return str(screenshot_path)

	async def get_screenshot(self, screenshot_path: str) -> Image.Image:
		"""Load screenshot from disk path and return as Image.Image"""
		if not screenshot_path:
			return None

		path = Path(screenshot_path)
		if not path.exists():
			return None

		return Image.open(path)

	async def draw_cursor(self, img: Image.Image, x: int, y: int, size: int = 32) -> Image.Image:
		"""
		Draw a Mac-style cursor on the screenshot using SVG.
		
		Args:
			img: Image.Image
			x: X coordinate of the cursor
			y: Y coordinate of the cursor
			size: Size of the cursor
			
		Returns:
			Image.Image: The screenshot with the cursor drawn on it
		"""	
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
		cursor_x = max(0, min(cursor_x, img.width - size))
		cursor_y = max(0, min(cursor_y, img.height - size))
		
		# Paste cursor onto screenshot
		if cursor_img.mode == 'RGBA':
			img.paste(cursor_img, (cursor_x, cursor_y), cursor_img)
		else:
			img.paste(cursor_img, (cursor_x, cursor_y))
   
		return img

	async def draw_path(self, img: Image.Image, path: list[list[int]], arrow_size: int = 16) -> Image.Image:
		"""
		Draw a path on the screenshot with arrows showing direction.
		
		Args:
			img: Image.Image
			path: List of [x, y] coordinates representing the path
			arrow_size: Size of the direction arrows
			
		Returns:
			Image.Image: The screenshot with the path drawn on it
		"""
		import math
		
		draw = ImageDraw.Draw(img)
		
		if len(path) < 2:
			return img
		
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

		return img

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

	async def draw_scroll(self, img: Image.Image, x: int, y: int, scroll_x: int, scroll_y: int) -> Image.Image:
		"""
		Draw a scroll operation on the screenshot.
		
		Args:
			img: Image.Image
			x: X coordinate where scroll started
			y: Y coordinate where scroll started
			scroll_x: X scroll delta (positive = right, negative = left)
			scroll_y: Y scroll delta (positive = down, negative = up)
			
		Returns:
			Image.Image: The screenshot with the scroll drawn on it
		"""
		import math
		
		# Load the screenshot
		draw = ImageDraw.Draw(img)
		
		# Calculate scroll direction and magnitude
		scroll_magnitude = math.sqrt(scroll_x * scroll_x + scroll_y * scroll_y)
		if scroll_magnitude == 0:
			return img
		
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
		
		return img