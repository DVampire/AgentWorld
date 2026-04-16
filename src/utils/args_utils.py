"""Utility functions for parsing tool arguments."""

import json
import re
from typing import Dict, Any

import dirtyjson

# from src.logger import logger


def parse_tool_args(args_str: str) -> Dict[str, Any]:
    """Parse tool arguments string to dictionary with multiple fallback strategies.
    
    Handles cases where JSON contains unescaped backslashes (e.g., LaTeX like \\frac).
    
    Args:
        args_str: Raw JSON string from LLM output
        
    Returns:
        Parsed dictionary or empty dict if all parsing attempts fail
    """
    if not args_str:
        return {}
    
    # Strategy 1: Try dirtyjson first (handles some malformed JSON)
    try:
        return dirtyjson.loads(args_str)
    except (dirtyjson.Error, ValueError, TypeError) as e:
        pass
    
    # Strategy 2: Try standard json
    try:
        return json.loads(args_str)
    except json.JSONDecodeError as e:
        pass
    
    # Strategy 3: Fix unescaped backslashes and try again
    # Replace single backslashes (not followed by valid escape chars) with double backslashes
    try:
        # JSON valid escape sequences: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
        # Replace \ not followed by these with \\
        fixed_str = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', args_str)
        return json.loads(fixed_str)
    except (json.JSONDecodeError, re.error) as e:
        pass
    
    # Strategy 4: Try to extract key-value pairs using regex as last resort
    try:
        result = {}
        # Match "key": "value" or "key": number patterns
        pattern = r'"(\w+)"\s*:\s*(?:"((?:[^"\\]|\\.)*)"|(\d+(?:\.\d+)?))'
        matches = re.findall(pattern, args_str)
        for key, str_val, num_val in matches:
            if str_val:
                # Unescape the string value
                result[key] = str_val.encode().decode('unicode_escape', errors='replace')
            elif num_val:
                result[key] = int(num_val) if '.' not in num_val else float(num_val)
        if result:
            return result
    except Exception as e:
        pass
    
    # All strategies failed
    return {}
