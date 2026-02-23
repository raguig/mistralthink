import base64
import re
from io import BytesIO


def encode_image(image_pil):
    buffered = BytesIO()
    image_pil.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def normalize_reply_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "\n".join([p for p in parts if p]).strip() or "(No text response)"
    return str(content)


def is_math_query(text):
    if not text:
        return False
    t = text.lower()
    math_keywords = [
        "integral",
        "differentiate",
        "derivative",
        "equation",
        "solve",
        "simplify",
        "factor",
        "limit",
        "matrix",
        "algebra",
        "calculate",
        "compute",
    ]
    if any(k in t for k in math_keywords):
        return True
    return bool(re.search(r"[\d\)\]]\s*[\+\-\*/\^]\s*[\d\(\[]", t))

