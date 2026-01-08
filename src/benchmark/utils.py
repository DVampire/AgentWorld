
def clean_text(text: str) -> str:
    if not text: 
        return ""
    text = str(text).strip().replace(",", "").replace("$", "")
    try:
        return str(int(float(text)))
    except (ValueError, TypeError):
        return text
