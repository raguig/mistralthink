import json
import subprocess

from config import SANDBOX_TIMEOUT_SECONDS


def run_code_in_sandbox(user_code, timeout_seconds=SANDBOX_TIMEOUT_SECONDS):
    payload = json.dumps(user_code)
    sandbox_script = f"""
import io, contextlib, traceback, base64, json, os, time, pathlib
user_code = {payload}
sanitized_code = user_code.replace("plt.show()", "").replace("matplotlib.pyplot.show()", "")
stdout_buffer = io.StringIO()
local = {{"os": os, "time": time, "pathlib": pathlib}}
allowed_roots = {{"math","statistics","numpy","matplotlib","seaborn","pandas","os","time","pathlib"}}
real_import = __import__
def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in allowed_roots:
        raise ImportError(f"Import '{{name}}' is blocked in sandbox.")
    return real_import(name, globals, locals, fromlist, level)
safe_builtins = {{
    "__import__": safe_import,
    "print": print,
    "range": range,
    "len": len,
    "sum": sum,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "int": int,
    "float": float,
    "str": str,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "enumerate": enumerate,
    "zip": zip,
}}
try:
    with contextlib.redirect_stdout(stdout_buffer):
        exec(sanitized_code, {{"__builtins__": safe_builtins}}, local)
    output = stdout_buffer.getvalue().strip() or local.get("result", "Executed (no output)")
    plot_base64 = None
    try:
        import matplotlib.pyplot as plt
        if plt.get_fignums():
            buf = io.BytesIO()
            plt.gcf().savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            plot_base64 = base64.b64encode(buf.read()).decode("utf-8")
    except Exception:
        pass
    finally:
        try:
            plt.close("all")
        except Exception:
            pass
    print("SANDBOX_RESULT:" + json.dumps({{"ok": True, "text": "Code output:\\n" + str(output), "plot_base64": plot_base64}}))
except Exception:
    print("SANDBOX_RESULT:" + json.dumps({{"ok": False, "text": "Code error:\\n" + traceback.format_exc(limit=2), "plot_base64": None}}))
"""
    try:
        completed = subprocess.run(
            ["py", "-3.11", "-c", sandbox_script],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return "Code error: sandbox timeout after {0} seconds.".format(timeout_seconds), None
    except Exception as e:
        return f"Code error: sandbox launch failed: {e}", None

    lines = (completed.stdout or "").splitlines()
    for line in reversed(lines):
        if line.startswith("SANDBOX_RESULT:"):
            try:
                result = json.loads(line[len("SANDBOX_RESULT:"):])
                return result.get("text", "Code error: unknown sandbox output."), result.get("plot_base64")
            except Exception:
                break
    if completed.stderr:
        return "Code error:\n" + completed.stderr.strip(), None
    return "Code error: sandbox terminated without parsable output.", None
