import contextlib
import io
import json
import math
import re

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

from sandbox import run_code_in_sandbox
from utils import is_math_query


tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression safely. Use for calculations, especially from images/charts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression, e.g. 'sin(pi/2) + 3 * 4'"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for facts, current events, or info not in your knowledge.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "code_interpreter",
            "description": "Execute simple Python code and return output. Use for plotting, math, or analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code snippet"}
                },
                "required": ["code"]
            }
        }
    }
]


def infer_required_tools(query_text: str):
    t = (query_text or "").lower()
    required = []

    asks_web_search = any(
        k in t for k in [
            "latest",
            "recent",
            "news",
            "current",
            "today",
            "search the web",
            "web search",
            "look up",
            "online",
        ]
    )
    asks_code_execution = any(
        k in t for k in [
            "plot",
            "visualize",
            "graph",
            "chart",
            "run code",
            "write code",
            "python code",
            "script",
            "execute code",
        ]
    )

    if asks_web_search:
        required.append("web_search")
    if asks_code_execution:
        required.append("code_interpreter")
    if is_math_query(t):
        required.append("calculator")
    return required


def infer_required_tools_from_plan(plan_text: str):
    p = (plan_text or "").lower()
    required = []
    if "web_search" in p:
        required.append("web_search")
    if "code_interpreter" in p:
        required.append("code_interpreter")
    if "calculator" in p:
        required.append("calculator")
    return required


def execute_tool_by_name_and_args(name, raw_args):
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        if not isinstance(args, dict):
            return "Invalid tool arguments.", None
    except Exception:
        return "Invalid tool arguments.", None

    try:
        if name == "calculator":
            expression = str(args.get("expression", ""))
            expression = expression.replace("^", "**")
            safe_dict = {
                k: v
                for k, v in math.__dict__.items()
                if (not k.startswith("_") and callable(v)) or isinstance(v, (int, float))
            }
            safe_dict["__builtins__"] = {}
            safe_dict["round"] = round
            result = eval(expression, safe_dict)
            return f"Calculation result: {result}", None

        if name == "web_search":
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                with DDGS() as ddgs:
                    results = [r for r in ddgs.text(args["query"], max_results=3)]
            if not results:
                return f"web_search(query={args.get('query', '')}) -> No relevant results found.", None
            summaries = [f"- {r['title']}: {r['body'][:300]}... Source: {r['href']}" for r in results]
            return f"web_search(query={args.get('query', '')}) results:\n" + "\n".join(summaries), None

        if name == "code_interpreter":
            user_code = args.get("code", "")
            if not isinstance(user_code, str) or not user_code.strip():
                return "Code error: missing 'code' string.", None
            return run_code_in_sandbox(user_code)

        return "Unknown tool.", None
    except Exception as e:
        return f"Tool execution failed: {str(e)}", None
