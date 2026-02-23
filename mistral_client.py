import httpx
from mistralai import Mistral

from config import api_key
from utils import normalize_reply_content


client = Mistral(api_key=api_key)


def safe_chat_complete(**kwargs):
    try:
        return client.chat.complete(**kwargs)
    except (httpx.ConnectTimeout, httpx.ReadTimeout, TimeoutError):
        raise RuntimeError("Network timeout while contacting Mistral API. Please retry in a few seconds.")
    except Exception as e:
        raise RuntimeError(f"Mistral API request failed: {e}")


def safe_chat_stream(**kwargs):
    try:
        return client.chat.stream(**kwargs)
    except (httpx.ConnectTimeout, httpx.ReadTimeout, TimeoutError):
        raise RuntimeError("Network timeout while contacting Mistral API. Please retry in a few seconds.")
    except Exception as e:
        raise RuntimeError(f"Mistral API request failed: {e}")


def collect_streamed_response(stream):
    content_parts = []
    tool_calls_by_index = {}

    for chunk in stream:
        data = getattr(chunk, "data", chunk)
        choices = getattr(data, "choices", None) or []
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        if not delta:
            continue

        delta_content = getattr(delta, "content", None)
        if delta_content:
            content_parts.append(normalize_reply_content(delta_content))

        delta_tool_calls = getattr(delta, "tool_calls", None) or []
        for tc in delta_tool_calls:
            idx = getattr(tc, "index", None)
            if idx is None:
                idx = len(tool_calls_by_index)
            entry = tool_calls_by_index.setdefault(
                idx, {"id": "", "name": "", "arguments": ""}
            )
            tc_id = getattr(tc, "id", None)
            if tc_id:
                entry["id"] = tc_id
            fn = getattr(tc, "function", None)
            if fn:
                fn_name = getattr(fn, "name", None)
                if fn_name:
                    entry["name"] = fn_name
                fn_args = getattr(fn, "arguments", None)
                if fn_args:
                    entry["arguments"] += fn_args

    normalized_calls = []
    for idx in sorted(tool_calls_by_index.keys()):
        call = tool_calls_by_index[idx]
        normalized_calls.append(
            {
                "id": call["id"] or f"tool_call_{idx}",
                "name": call["name"],
                "arguments": call["arguments"] or "{}",
            }
        )

    full_text = "".join([p for p in content_parts if p]).strip()
    if not full_text:
        full_text = ""
    return full_text, normalized_calls

