_ACTION_DESCRIPTION="""The operation to perform. 

Available operations:
1. key: Press a key or key combination on the keyboard.
    - Supports the `key` syntax of xdotool.
    - Examples: "a", "Return", "alt+Tab", "ctrl+s", "Up", "KP_0" (for the numeric keypad 0 key).
2. hold_key: Hold a key or key combination on the keyboard.
    - Hold down one or more keys for a specified amount of time (in seconds). Supports the same syntax as `key`
3. type: Type a string of text on the keyboard.
4. cursor_position: Get the current (x, y) pixel coordinates of the cursor on the screen.
5. mouse_move: Move the cursor to the specified (x, y) pixel coordinates on the screen.
6. left_mouse_down: Press the left mouse button.
7. left_mouse_up: Release the left mouse button.
8. left_click: Click the left mouse button at the specified (x, y) pixel coordinates on the screen. You can also use the `text` parameter to include key combinations to hold during the click.
9. left_click_drag: Click and drag the cursor from `start_coordinate` to the specified (x, y) pixel coordinates on the screen.
10. right_click: Right-click at the specified (x, y) pixel coordinates on the screen.
11. middle_click: Middle-click at the specified (x, y) pixel coordinates on the screen.
12. double_click: Double-click the left mouse button at the specified (x, y) pixel coordinates on the screen.
13. triple_click: Triple-click the left mouse button at the specified (x, y) pixel coordinates on the screen.
14. scroll: Scroll the screen at the specified (x, y) pixel coordinates by a given number of wheel ticks in the specified direction. Do not use PageUp/PageDown to scroll.
15. wait: Wait for a specified amount of time (in seconds).
16. screenshot: Take a screenshot of the screen.
"""