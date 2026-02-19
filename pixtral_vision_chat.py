import os
import base64
import warnings
from io import BytesIO
from PIL import Image
import gradio as gr
from mistralai import Mistral

import json
import math
import re
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

from dotenv import load_dotenv




load_dotenv()
api_key = (os.getenv("MISTRAL_API_KEY") or "").strip().strip('"').strip("'")
if not api_key:
    raise ValueError("MISTRAL_API_KEY not found in .env")

warnings.filterwarnings(
    "ignore",
    message=".*HTTP_422_UNPROCESSABLE_ENTITY.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*duckduckgo_search.*has been renamed to `ddgs`.*",
    category=RuntimeWarning,
)

client = Mistral(api_key=api_key)

MODEL = "pixtral-large-latest"


# Tool schemas — Mistral expects OpenAI-compatible format
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

def execute_tool(tool_call):
    """Run the tool and return string result"""
    if not tool_call or not tool_call.function:
        return "No tool call found."

    name = tool_call.function.name
    try:
        raw_args = tool_call.function.arguments
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        if not isinstance(args, dict):
            return "Invalid tool arguments."
    except Exception:
        return "Invalid tool arguments."

    try:
        if name == "calculator":
            # Safe eval with math only
            safe_dict = {"__builtins__": {}, **math.__dict__}
            result = eval(args["expression"], safe_dict)
            return f"Calculation result: {result}"

        elif name == "web_search":
            with DDGS() as ddgs:
                results = [r for r in ddgs.text(args["query"], max_results=3)]
            if not results:
                return "No relevant results found."
            summaries = [f"[{r['title']}] {r['body'][:300]}... ({r['href']})" for r in results]
            return "\n\n".join(summaries)

        elif name == "code_interpreter":
            # Very basic exec — in production use restricted env/sandbox
            local = {}
            try:
                exec(args["code"], {"__builtins__": {}}, local)
                output = local.get("result", local.get("print_output", "Executed (no 'result' var)"))
                return f"Code output:\n{output}"
            except Exception as e:
                return f"Code error: {str(e)}"

        else:
            return "Unknown tool."
    except Exception as e:
        return f"Tool execution failed: {str(e)}"


def encode_image(image_pil):
    buffered = BytesIO()
    image_pil.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

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
        "integral", "differentiate", "derivative", "equation", "solve",
        "simplify", "factor", "limit", "matrix", "algebra", "calculate", "compute"
    ]
    if any(k in t for k in math_keywords):
        return True
    return bool(re.search(r"[\d\)\]]\s*[\+\-\*/\^]\s*[\d\(\[]", t))


def multimodal_chat(image, text, api_history):
    messages = api_history.copy() if api_history else []
    reply = "No final response generated."

    current_content = [{"type": "text", "text": text or "Describe this image in detail."}]
    if image is not None:
        base64_img = encode_image(image)
        current_content.append({
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{base64_img}"
        })

    messages.append({"role": "user", "content": current_content})

    force_calculator_for_math = is_math_query(text)
    tool_choice = (
        {"type": "function", "function": {"name": "calculator"}}
        if force_calculator_for_math
        else "auto"
    )
    max_tool_rounds = 3  # Prevent infinite loops
    for round in range(max_tool_rounds):
        try:
            response = client.chat.complete(
                model=MODEL,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=1024,
                temperature=0.7
            )
            choice = response.choices[0]

            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                # Add assistant tool-call message once, then add one tool result per call.
                messages.append(choice.message.model_dump(exclude_none=True))
                for tool_call in choice.message.tool_calls:
                    tool_result = execute_tool(tool_call)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": tool_result
                    })
            else:
                # Normal reply — done
                reply = normalize_reply_content(choice.message.content)
                messages.append({"role": "assistant", "content": reply})
                break

        except Exception as e:
            reply = f"API Error in round {round+1}: {str(e)}"
            messages.append({"role": "assistant", "content": reply})
            break

    # If every round used tools, run one final pass to produce plain assistant text.
    if reply == "No final response generated.":
        try:
            final_response = client.chat.complete(
                model=MODEL,
                messages=messages,
                max_tokens=1024,
                temperature=0.7
            )
            final_choice = final_response.choices[0]
            reply = normalize_reply_content(final_choice.message.content)
            messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            reply = f"API Error after tools: {str(e)}"
            messages.append({"role": "assistant", "content": reply})

    return messages, reply

# Gradio UI
with gr.Blocks(title="Pixtral Vision Chat – Phase 1 Fixed") as demo:
    gr.Markdown("# Pixtral Multimodal Agent – Phase 1\nUpload image + ask anything about it!")
    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(placeholder="Ask about the image (e.g., 'What trends do you see here?')", label="Your question")
    img_input = gr.Image(type="pil", label="Upload Image (JPEG/PNG)")
    clear = gr.Button("Clear Conversation")

    api_state = gr.State([])
    chat_state = gr.State([])

    def respond(message, image, api_history, ui_history):
        new_api_messages, reply = multimodal_chat(image, message, api_history)
        user_text = message or "Describe this image in detail."
        if image is not None:
            user_text = f"{user_text}\n[Image attached]"

        previous_len = len(api_history or [])
        turn_messages = new_api_messages[previous_len:]
        used_tools = []
        for msg_item in turn_messages:
            if isinstance(msg_item, dict) and msg_item.get("role") == "tool" and msg_item.get("name"):
                used_tools.append(msg_item["name"])
        tool_list = ", ".join(dict.fromkeys(used_tools))
        assistant_text = f"{reply}\n\nTools used: {tool_list}" if tool_list else reply
        new_ui_history = (ui_history or []) + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
        return "", new_api_messages, new_ui_history, new_ui_history

    msg.submit(
        respond,
        inputs=[msg, img_input, api_state, chat_state],
        outputs=[msg, api_state, chat_state, chatbot]
    )
    
    clear.click(
        lambda: ("", [], [], []),
        None,
        [msg, api_state, chat_state, chatbot]
    )

demo.launch(share=False)
