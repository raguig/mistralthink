import base64
import logging
import math
import re
import warnings
from io import BytesIO

import gradio as gr
from PIL import Image
from langchain_core.messages import HumanMessage, ToolMessage

from agent import app
from tools import infer_required_tools
from utils import encode_image, normalize_reply_content


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
warnings.filterwarnings(
    "ignore",
    message=".*FigureCanvasAgg is non-interactive, and thus cannot be shown.*",
    category=UserWarning,
)
logging.getLogger("primp").setLevel(logging.ERROR)
logging.getLogger("ddgs").setLevel(logging.ERROR)


with gr.Blocks(title="Pixtral Multimodal Agent") as demo:
    gr.Markdown("# Pixtral Multimodal Agent\nUpload image + ask anything about it!")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=500, label="Conversation")
        with gr.Column(scale=1):
            plan_display = gr.Textbox(label="Current Agent Plan", interactive=False, lines=10)
            summary_display = gr.Textbox(label="Conversation Summary", interactive=False, lines=5)
            plot_display = gr.Image(label="Generated Plot", type="pil")

    msg = gr.Textbox(placeholder="Ask about the image (e.g., 'What trends do you see here?')", label="Your question")
    img_input = gr.Image(type="pil", label="Upload Image (JPEG/PNG)")
    clear = gr.Button("Clear Conversation")

    api_state = gr.State([])
    chat_state = gr.State([])
    summary_state = gr.State("")
    image_state = gr.State("")

    def respond(message, image, api_history, ui_history, running_summary, stored_image):
        current_image = encode_image(image) if image is not None else (stored_image or "")
        original_required_tools = infer_required_tools(message or "")
        inputs = {
            "messages": (api_history or []) + [HumanMessage(content=message or "")],
            "summary": running_summary or "",
            "image_data": current_image,
            "plan": "",
            "needs_retry": False,
            "retry_count": 0,
            "required_tools": original_required_tools,
        }

        try:
            result = app.invoke(inputs, config={"recursion_limit": 80})
        except Exception as e:
            error_reply = f"Temporary failure: {e}"
            new_ui_history = (ui_history or []) + [
                {"role": "user", "content": message or ""},
                {"role": "assistant", "content": error_reply},
            ]
            yield "", api_history or [], new_ui_history, new_ui_history, running_summary or "", "", None, running_summary or "", current_image
            return

        final_reply = normalize_reply_content(result["messages"][-1].content)

        final_reply = re.sub(r"\n?\[Critique:.*?\)]", "", final_reply, flags=re.DOTALL)
        final_reply = re.sub(r"\n?Stopped after retry limit.*", "", final_reply, flags=re.DOTALL)
        final_reply = re.sub(r'\[\{\"name\".*?\}\]', "", final_reply, flags=re.DOTALL)
        final_reply = re.sub(r'\[?\{\"name\": \"code_interpreter\".*', "", final_reply, flags=re.DOTALL)
        final_reply = re.sub(r"^\s*\*{0,2}revised answer:?\*{0,2}\s*", "", final_reply, flags=re.IGNORECASE)
        final_reply = re.sub(r"!\[[^\]]*\]\(attachment:[^)]+\)", "", final_reply, flags=re.IGNORECASE)
        final_reply = re.sub(r"\n{3,}", "\n\n", final_reply).strip()
        new_api_history = result["messages"]
        if not final_reply.strip():
            for msg_obj in reversed(new_api_history):
                if isinstance(msg_obj, ToolMessage):
                    tool_text = normalize_reply_content(msg_obj.content).strip()
                    if tool_text:
                        final_reply = tool_text
                        break
        if not final_reply.strip():
            final_reply = "I couldn't generate a final response, but I can retry if you send the same request again."
        new_summary = result.get("summary", running_summary or "")
        base_ui_history = (ui_history or []) + [{"role": "user", "content": message or ""}]

        plot_image = None
        for msg_obj in reversed(new_api_history):
            if isinstance(msg_obj, ToolMessage):
                plot_b64 = msg_obj.additional_kwargs.get("plot_base64")
                if plot_b64:
                    try:
                        img_data = base64.b64decode(plot_b64)
                        with BytesIO(img_data) as bio:
                            plot_image = Image.open(bio).copy()
                        break
                    except Exception:
                        pass

        tokens = re.split(r"(\s+)", final_reply)
        chunk_count = min(80, max(1, len(tokens)))
        step = max(1, math.ceil(len(tokens) / chunk_count))

        for i in range(step, len(tokens) + step, step):
            current_reply = "".join(tokens[:i]).strip()
            streaming_ui_history = base_ui_history + [{"role": "assistant", "content": current_reply}]
            yield (
                "",
                new_api_history,
                streaming_ui_history,
                streaming_ui_history,
                new_summary,
                result.get("plan", ""),
                None,
                new_summary,
                current_image,
            )

        final_ui_history = base_ui_history + [{"role": "assistant", "content": final_reply}]
        yield (
            "",
            new_api_history,
            final_ui_history,
            final_ui_history,
            new_summary,
            result.get("plan", ""),
            plot_image,
            new_summary,
            current_image,
        )

    msg.submit(
        respond,
        inputs=[msg, img_input, api_state, chat_state, summary_state, image_state],
        outputs=[msg, api_state, chat_state, chatbot, summary_state, plan_display, plot_display, summary_display, image_state],
    )

    clear.click(
        lambda: ("", [], [], [], "", "", None, "", ""),
        None,
        [msg, api_state, chat_state, chatbot, summary_state, plan_display, plot_display, summary_display, image_state],
    )


if __name__ == "__main__":
    demo.launch(share=False)
