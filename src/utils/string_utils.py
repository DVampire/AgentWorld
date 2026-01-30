import hashlib
import uuid
from datetime import datetime

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