"""Keyboard class for keyboard operations."""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
	from cdp_use.cdp.input.commands import DispatchKeyEventParameters

	from src.environments.browser.browser.session import BrowserSession


class Keyboard:
	"""Keyboard operations for a target."""

	def __init__(self, browser_session: 'BrowserSession', session_id: Optional[str] = None, target_id: Optional[str] = None):
		self._browser_session = browser_session
		self._client = browser_session.cdp_client
		self._session_id = session_id
		self._target_id = target_id

	async def type(self, text: str, delay: int = 0) -> None:
		"""Type text with optional delay between keystrokes."""
		if not self._session_id:
			raise RuntimeError('Session ID is required for keyboard operations')

		# Activate the target first (critical for CDP calls to work)
		if self._target_id:
			try:
				await self._client.send.Target.activateTarget(params={'targetId': self._target_id})
			except Exception:
				pass

		# Type each character
		for char in text:
			# Key down
			down_params: 'DispatchKeyEventParameters' = {
				'type': 'keyDown',
				'key': char,
				'text': char,
			}
			await self._client.send.Input.dispatchKeyEvent(
				down_params,
				session_id=self._session_id,
			)

			# Key up
			up_params: 'DispatchKeyEventParameters' = {
				'type': 'keyUp',
				'key': char,
				'text': char,
			}
			await self._client.send.Input.dispatchKeyEvent(
				up_params,
				session_id=self._session_id,
			)

			# Add delay if specified
			if delay > 0:
				import asyncio
				await asyncio.sleep(delay / 1000.0)  # Convert ms to seconds

	async def press(self, key: str, modifiers: Optional[int] = None) -> None:
		"""Press a key with optional modifiers."""
		if not self._session_id:
			raise RuntimeError('Session ID is required for keyboard operations')

		# Activate the target first (critical for CDP calls to work)
		if self._target_id:
			try:
				await self._client.send.Target.activateTarget(params={'targetId': self._target_id})
			except Exception:
				pass

		# Handle special keys
		key_code = self._get_key_code(key)
		text = self._get_key_text(key)

		# Key down
		down_params: 'DispatchKeyEventParameters' = {
			'type': 'keyDown',
			'key': key,
			'code': key_code,
			'text': text,
		}
		if modifiers:
			down_params['modifiers'] = modifiers

		await self._client.send.Input.dispatchKeyEvent(
			down_params,
			session_id=self._session_id,
		)

		# Key up
		up_params: 'DispatchKeyEventParameters' = {
			'type': 'keyUp',
			'key': key,
			'code': key_code,
			'text': text,
		}
		if modifiers:
			up_params['modifiers'] = modifiers

		await self._client.send.Input.dispatchKeyEvent(
			up_params,
			session_id=self._session_id,
		)

	async def down(self, key: str, modifiers: Optional[int] = None) -> None:
		"""Press a key down (without releasing)."""
		if not self._session_id:
			raise RuntimeError('Session ID is required for keyboard operations')

		# Activate the target first (critical for CDP calls to work)
		if self._target_id:
			try:
				await self._client.send.Target.activateTarget(params={'targetId': self._target_id})
			except Exception:
				pass

		key_code = self._get_key_code(key)
		text = self._get_key_text(key)

		params: 'DispatchKeyEventParameters' = {
			'type': 'keyDown',
			'key': key,
			'code': key_code,
			'text': text,
		}
		if modifiers:
			params['modifiers'] = modifiers

		await self._client.send.Input.dispatchKeyEvent(
			params,
			session_id=self._session_id,
		)

	async def up(self, key: str, modifiers: Optional[int] = None) -> None:
		"""Release a key."""
		if not self._session_id:
			raise RuntimeError('Session ID is required for keyboard operations')

		# Activate the target first (critical for CDP calls to work)
		if self._target_id:
			try:
				await self._client.send.Target.activateTarget(params={'targetId': self._target_id})
			except Exception:
				pass

		key_code = self._get_key_code(key)
		text = self._get_key_text(key)

		params: 'DispatchKeyEventParameters' = {
			'type': 'keyUp',
			'key': key,
			'code': key_code,
			'text': text,
		}
		if modifiers:
			params['modifiers'] = modifiers

		await self._client.send.Input.dispatchKeyEvent(
			params,
			session_id=self._session_id,
		)

	def _get_key_code(self, key: str) -> str:
		"""Get the key code for a key."""
		# Common key mappings
		key_mappings = {
			'Enter': 'Enter',
			'Return': 'Enter',
			'Tab': 'Tab',
			'Space': 'Space',
			'Backspace': 'Backspace',
			'Delete': 'Delete',
			'Escape': 'Escape',
			'ArrowUp': 'ArrowUp',
			'ArrowDown': 'ArrowDown',
			'ArrowLeft': 'ArrowLeft',
			'ArrowRight': 'ArrowRight',
			'Home': 'Home',
			'End': 'End',
			'PageUp': 'PageUp',
			'PageDown': 'PageDown',
			'F1': 'F1',
			'F2': 'F2',
			'F3': 'F3',
			'F4': 'F4',
			'F5': 'F5',
			'F6': 'F6',
			'F7': 'F7',
			'F8': 'F8',
			'F9': 'F9',
			'F10': 'F10',
			'F11': 'F11',
			'F12': 'F12',
			'Control': 'ControlLeft',
			'Ctrl': 'ControlLeft',
			'Alt': 'AltLeft',
			'Shift': 'ShiftLeft',
			'Meta': 'MetaLeft',
			'Cmd': 'MetaLeft',
		}
		return key_mappings.get(key, key)

	def _get_key_text(self, key: str) -> str:
		"""Get the text representation for a key."""
		# For special keys, return empty string
		special_keys = {
			'Enter', 'Return', 'Tab', 'Space', 'Backspace', 'Delete', 'Escape',
			'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Home', 'End',
			'PageUp', 'PageDown', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7',
			'F8', 'F9', 'F10', 'F11', 'F12', 'Control', 'Ctrl', 'Alt', 'Shift',
			'Meta', 'Cmd'
		}
		
		if key in special_keys:
			return ''
		return key
