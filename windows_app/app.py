import sys
import os
import time
import threading
import torch
import gradio as gr
from PIL import Image
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from transformers import TextIteratorStreamer

# Global state
tokenizer = None
model = None
image_processor = None
current_model_name = None
is_generating_live = False
last_live_output = ""
performance_history = deque(maxlen=30)
frame_skip_counter = 0
cached_prompt_ids = None
cached_prompt_text = None

MODELS = {
    "Stage 2 (0.5B)": "../checkpoints/llava-fastvithd_0.5b_stage2",
    "Stage 3 (0.5B)": "../checkpoints/llava-fastvithd_0.5b_stage3"
}

PRESET_PROMPTS = {
    "Describe": "Describe what you see briefly.",
    "Count Objects": "Count the main objects visible.",
    "Read Text": "Read any text visible in the image.",
    "Identify": "What is the main subject?",
    "Action": "What action is happening?",
    "Custom": ""
}


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_fn(model_choice):
    global tokenizer, model, image_processor, current_model_name
    global cached_prompt_ids, cached_prompt_text

    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), MODELS[model_choice]))

    if current_model_name == model_choice:
        return f"✅ {model_choice} already loaded"

    try:
        if model is not None:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        cached_prompt_ids = None
        cached_prompt_text = None
        device = get_device()

        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),
            device=device
        )

        model.eval()
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        if tokenizer.pad_token_id is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id

        current_model_name = model_choice
        return f"✅ {model_choice} on {device.upper()}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def prepare_prompt(prompt):
    global cached_prompt_ids, cached_prompt_text

    if cached_prompt_text == prompt and cached_prompt_ids is not None:
        return cached_prompt_ids.clone()

    if model.config.mm_use_im_start_end:
        qs = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{prompt}"
    else:
        qs = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"

    conv = conv_templates["qwen_2"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)

    input_ids = tokenizer_image_token(
        conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(model.device)

    cached_prompt_text = prompt
    cached_prompt_ids = input_ids
    return input_ids


def clean_output(text):
    for pattern in ["Answer:", "Response:", "Output:", "Description:"]:
        if pattern in text:
            text = text.split(pattern)[0].strip()

    if "\`\`\`" in text:
        text = text.split("\`\`\`")[0].strip()

    text = text.replace("**", "").replace("*", "")

    for delimiter in ['\n\n', '\n', '. ']:
        if delimiter in text:
            text = text.split(delimiter)[0].strip()
            if delimiter == '. ' and text:
                text += '.'
            break

    if text and text[-1] not in '.!?' and len(text) > 20:
        last_space = text.rfind(' ')
        if last_space > len(text) * 0.7:
            text = text[:last_space] + "..."

    if len(text) > 120:
        text = text[:120].rsplit(' ', 1)[0] + "..."

    return text


def get_stats():
    if not performance_history:
        return "No data yet"
    avg = sum(performance_history) / len(performance_history)
    return f"Avg: {avg:.0f}ms | Min: {min(performance_history):.0f}ms | Max: {max(performance_history):.0f}ms | ~FPS: {1000/avg:.1f}"


def live_inference(image, prompt, frame_skip):
    global is_generating_live, last_live_output, frame_skip_counter, performance_history

    if image is None:
        return last_live_output, "⏸️ Waiting...", get_stats()

    if model is None:
        return "⚠️ Load a model first", "❌ No model", ""

    frame_skip_counter += 1
    if frame_skip_counter < frame_skip:
        return last_live_output, f"⏭️ Skip {frame_skip_counter}/{frame_skip}", get_stats()
    frame_skip_counter = 0

    if is_generating_live:
        return last_live_output, "⏳ Processing...", get_stats()

    is_generating_live = True

    try:
        start = time.time()
        input_ids = prepare_prompt(prompt)
        image_tensor = process_images([image], image_processor, model.config)[0]

        with torch.inference_mode():
            output_ids = model.generate(
                inputs=input_ids,
                images=image_tensor.unsqueeze(0).half(),
                image_sizes=[image.size],
                do_sample=False,
                max_new_tokens=32,
                min_new_tokens=3,
                use_cache=True,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                eos_token_id=tokenizer.eos_token_id,
            )

        output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        output = clean_output(output)

        ms = (time.time() - start) * 1000
        performance_history.append(ms)

        if output:
            last_live_output = output

        return output or "[No output]", f"✅ {ms:.0f}ms", get_stats()

    except Exception as e:
        return f"Error: {str(e)[:50]}", "❌ Error", get_stats()
    finally:
        is_generating_live = False


def chat(message, history, image, temperature, top_p):
    if model is None:
        yield "Load a model first.", "No model"
        return
    if image is None:
        yield "Upload an image.", "No image"
        return

    if model.config.mm_use_im_start_end:
        qs = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{message}"
    else:
        qs = f"{DEFAULT_IMAGE_TOKEN}\n{message}"

    conv = conv_templates["qwen_2"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)

    input_ids = tokenizer_image_token(
        conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(model.device)

    image_tensor = process_images([image], image_processor, model.config)[0]
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        inputs=input_ids,
        images=image_tensor.unsqueeze(0).half(),
        image_sizes=[image.size],
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        top_p=top_p if temperature > 0 else None,
        max_new_tokens=64,
        streamer=streamer,
        use_cache=True,
        repetition_penalty=1.1,
    )

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    start = time.time()
    thread.start()

    text = ""
    first_token = None
    tokens = 0

    for chunk in streamer:
        if first_token is None:
            first_token = time.time()
        text += chunk
        tokens += 1
        elapsed = time.time() - start
        ttft = (first_token - start) * 1000 if first_token else 0
        yield text, f"TTFT: {ttft:.0f}ms | {tokens/elapsed:.1f} tok/s | {tokens} tokens"

    thread.join()


def reset_stats():
    performance_history.clear()
    return "Stats reset"


# UI
with gr.Blocks(title="FastVLM", theme=gr.themes.Default(primary_hue="orange")) as demo:
    gr.Markdown("# 🚀 FastVLM Inference")

    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(list(MODELS.keys()), label="Model", value="Stage 3 (0.5B)")
            load_btn = gr.Button("Load Model", variant="primary")
            load_status = gr.Textbox(label="Status", value="No model", interactive=False)

            temperature = gr.Slider(0, 1, value=0, step=0.1, label="Temperature")
            top_p = gr.Slider(0, 1, value=0.9, step=0.1, label="Top P")

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("💬 Chat"):
                    image_input = gr.Image(type="pil", label="Image")
                    chatbot = gr.Chatbot(height=250, type="tuples")
                    msg = gr.Textbox(label="Message", placeholder="Ask about the image...")
                    with gr.Row():
                        clear = gr.Button("Clear")
                        send = gr.Button("Send", variant="primary")
                    metrics = gr.Textbox(label="Metrics", interactive=False)

                with gr.Tab("📹 Live"):
                    with gr.Row():
                        webcam = gr.Image(sources=["webcam"], streaming=True, type="pil", label="Camera")
                        with gr.Column():
                            preset = gr.Dropdown(list(PRESET_PROMPTS.keys()), label="Preset", value="Describe")
                            live_prompt = gr.Textbox(label="Prompt", value="Describe what you see briefly.", lines=2)
                            frame_skip = gr.Slider(1, 10, value=3, step=1, label="Frame Skip")

                    output = gr.Textbox(label="Output", lines=2)
                    with gr.Row():
                        status = gr.Textbox(label="Status", value="⏸️ Waiting...", interactive=False)
                        reset_btn = gr.Button("Reset Stats")
                    stats = gr.Textbox(label="Stats", value="No data yet", interactive=False)

    # Events
    load_btn.click(load_model_fn, [model_dropdown], [load_status])
    preset.change(lambda p: PRESET_PROMPTS.get(p, ""), [preset], [live_prompt])
    reset_btn.click(reset_stats, outputs=[stats])

    webcam.stream(
        live_inference,
        [webcam, live_prompt, frame_skip],
        [output, status, stats],
        show_progress="hidden"
    )

    def user_msg(msg, history):
        return "", history + [[msg, None]]

    def bot_response(history, image, temp, top_p):
        if not history:
            return history, ""
        for text, met in chat(history[-1][0], history, image, temp, top_p):
            history[-1][1] = text
            yield history, met

    msg.submit(user_msg, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_response, [chatbot, image_input, temperature, top_p], [chatbot, metrics]
    )
    send.click(user_msg, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_response, [chatbot, image_input, temperature, top_p], [chatbot, metrics]
    )
    clear.click(lambda: (None, []), outputs=[image_input, chatbot])

if __name__ == "__main__":
    demo.queue().launch()