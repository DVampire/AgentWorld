import hashlib
import uuid
from datetime import datetime
from sympy import simplify
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor
)

transformations = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

def hash_text_sha256(text: str) -> str:
    hash_object = hashlib.sha256(text.encode())
    return hash_object.hexdigest()

def extract_boxed_content(text: str) -> str:
    """
    Extracts answers in \\boxed{}.
    """
    depth = 0
    start_pos = text.rfind(r"\boxed{")
    end_pos = -1
    if start_pos != -1:
        content = text[start_pos + len(r"\boxed{") :]
        for i, char in enumerate(content):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1

            if depth == -1:  # exit
                end_pos = i
                break

    if end_pos != -1:
        return content[:end_pos].strip()

    return "None"

def dedent(text: str) -> str:
    """
    Dedent the text and expand the tabs.
    """
    clean = "\n".join(line.strip() for line in text.splitlines())
    return clean

def generate_unique_id(prefix: str = "session") -> str:
    """Generate a unique id using timestamp and UUID."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{prefix}_{timestamp}_{unique_id}"

def _strip_latex_delimiters(text: str) -> str:
    """Strip LaTeX math delimiters: $$...$$, $...$, \\[...\\], \\(...\\)."""
    text = text.strip()
    if text.startswith("$$") and text.endswith("$$") and len(text) > 4:
        return text[2:-2].strip()
    if text.startswith("$") and text.endswith("$") and len(text) > 2:
        return text[1:-1].strip()
    if text.startswith(r"\[") and text.endswith(r"\]"):
        return text[2:-2].strip()
    if text.startswith(r"\(") and text.endswith(r"\)"):
        return text[2:-2].strip()
    return text


def _normalize_str(text: str) -> str:
    """Lightweight string normalization for fallback comparison."""
    import re
    text = _strip_latex_delimiters(text)
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip(' .,;:!?')
    return text


def _to_sympy(text: str):
    """Try to convert a string to a sympy expression. Returns None on failure."""
    # Try standard sympy parser first
    try:
        return parse_expr(text, transformations=transformations)
    except Exception:
        pass
    # Try LaTeX parser
    try:
        from sympy.parsing.latex import parse_latex
        return parse_latex(text)
    except Exception:
        pass
    return None


def is_same(a: str, b: str) -> bool:
    a = _strip_latex_delimiters(a)
    b = _strip_latex_delimiters(b)

    # Try sympy comparison — each side parsed independently
    expr_a = _to_sympy(a)
    expr_b = _to_sympy(b)
    if expr_a is not None and expr_b is not None:
        try:
            if simplify(expr_a - expr_b) == 0:
                return True
        except Exception:
            pass

    # Normalized string fallback
    return _normalize_str(a) == _normalize_str(b)