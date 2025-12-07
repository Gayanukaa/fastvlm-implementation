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
from transformers import TextIteratorStreamer

tokenizer = None
model = None
image_processor = None
context_len = None
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
    "Count Objects": "Count how many fingers i am holding up?",
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
    global tokenizer, model, image_processor, context_len, current_model_name
    global cached_prompt_ids, cached_prompt_text

    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), MODELS[model_choice]))

    if current_model_name == model_choice:
        return f"✅ Model {model_choice} already loaded."

    print(f"Loading {model_choice} from {model_path}...")
    try:
        if model is not None:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Clear prompt cache
        cached_prompt_ids = None
        cached_prompt_text = None

        model_name = get_model_name_from_path(model_path)
        device = get_device()

        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_name,
            device=device
        )

        model.eval()
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Set the pad token id for generation
        if tokenizer.pad_token_id is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id

        current_model_name = model_choice
        return f"✅ Loaded {model_choice} on {device.upper()}"
    except Exception as e:
        return f"❌ Error loading model: {str(e)}"

def prepare_prompt(prompt):
    """Prepare and cache the prompt tokens."""
    global cached_prompt_ids, cached_prompt_text

    # Use cached prompt if same
    if cached_prompt_text == prompt and cached_prompt_ids is not None:
        return cached_prompt_ids.clone()

    qs = prompt
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    from llava.conversation import conv_templates
    conv_mode = "qwen_2"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt_str = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_str, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

    # Cache for reuse
    cached_prompt_text = prompt
    cached_prompt_ids = input_ids

    return input_ids


def live_inference(image, prompt, temperature, top_p, frame_skip):
    global is_generating_live, last_live_output, model, tokenizer, image_processor
    global frame_skip_counter, performance_history

    if image is None:
        return last_live_output, "⏸️ Waiting for webcam...", get_performance_stats()

    if model is None:
        return "⚠️ Model not loaded.", "❌ Model not loaded", ""

    # Frame skipping for smoother experience
    frame_skip_counter += 1
    if frame_skip_counter < frame_skip:
        return last_live_output, f"⏭️ Skipping frame ({frame_skip_counter}/{frame_skip})", get_performance_stats()
    frame_skip_counter = 0

    # If busy, skip this frame
    if is_generating_live:
        return last_live_output, "⏳ Processing...", get_performance_stats()

    is_generating_live = True

    try:
        start_time = time.time()

        # Use cached prompt tokens
        input_ids = prepare_prompt(prompt)

        # Process Image with optimizations
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

        # Decode
        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        output_text = clean_live_output(output_text)

        # Track performance
        inference_time = (time.time() - start_time) * 1000
        performance_history.append(inference_time)

        print(f"[{inference_time:.0f}ms] {output_text}")

        if output_text:
            last_live_output = output_text
        else:
            output_text = "[No output]"

        status = f"✅ {inference_time:.0f}ms"
        return output_text, status, get_performance_stats()

    except Exception as e:
        print(f"Live inference error: {e}")
        return f"Error: {str(e)[:50]}", "❌ Error", get_performance_stats()
    finally:
        is_generating_live = False


def get_performance_stats():
    """Get performance statistics."""
    if not performance_history:
        return "**Stats:** No data yet"

    avg_time = sum(performance_history) / len(performance_history)
    min_time = min(performance_history)
    max_time = max(performance_history)
    fps = 1000 / avg_time if avg_time > 0 else 0

    return f"**Avg:** {avg_time:.0f}ms | **Min:** {min_time:.0f}ms | **Max:** {max_time:.0f}ms | **~FPS:** {fps:.1f}"


def clean_live_output(text):
    """Clean up model output for live video display."""
    # Remove common repetitive patterns
    for pattern in ["Answer:", "Response:", "Output:", "Description:"]:
        if pattern in text:
            text = text.split(pattern)[0].strip()

    # Remove code blocks
    if "```" in text:
        text = text.split("```")[0].strip()

    # Remove markdown artifacts
    text = text.replace("**", "").replace("*", "")

    # Take only first line/sentence
    for delimiter in ['\n\n', '\n', '. ']:
        if delimiter in text:
            parts = text.split(delimiter)
            text = parts[0].strip()
            if delimiter == '. ' and text:
                text += '.'
            break

    # Remove incomplete sentences at the end
    if text and text[-1] not in '.!?':
        last_space = text.rfind(' ')
        if last_space > len(text) * 0.7:  # Only trim if near the end
            text = text[:last_space] + "..."

    # Limit length
    if len(text) > 120:
        text = text[:120].rsplit(' ', 1)[0] + "..."

    return text


def update_prompt_from_preset(preset_choice):
    """Update prompt textbox based on preset selection."""
    return PRESET_PROMPTS.get(preset_choice, "")


def reset_performance():
    """Reset performance tracking."""
    global performance_history
    performance_history.clear()
    return "**Stats:** Reset"


def chat(message, history, image, temperature, top_p):
    global tokenizer, model, image_processor

    if model is None:
        yield "Please load a model first.", "**Status:** Model not loaded"
        return

    if image is None:
        yield "Please upload an image.", "**Status:** No image provided"
        return

    # Prepare prompt
    qs = message
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    # Prepare conversation
    from llava.conversation import conv_templates
    conv_mode = "qwen_2"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Tokenize
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

    # Process Image
    image_tensor = process_images([image], image_processor, model.config)[0]

    # Streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        inputs=input_ids,
        images=image_tensor.unsqueeze(0).half(),
        image_sizes=[image.size],
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        top_p=top_p if temperature > 0 else None,
        max_new_tokens=128,
        streamer=streamer,
        use_cache=True,
        repetition_penalty=1.1,
    )

    # Run generation in a separate thread
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)

    start_time = time.time()
    thread.start()

    generated_text = ""
    first_token_time = None
    token_count = 0

    for new_text in streamer:
        if first_token_time is None:
            first_token_time = time.time()

        generated_text += new_text
        token_count += 1

        # Calculate metrics
        current_time = time.time()
        ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
        total_time = current_time - start_time
        tps = token_count / total_time if total_time > 0 else 0

        metrics = f"**TTFT:** {ttft:.0f}ms | **Speed:** {tps:.1f} tok/s | **Tokens:** {token_count}"

        yield generated_text, metrics

    thread.join()

# UI Construction
with gr.Blocks(title="FastVLM Inference", theme=gr.themes.Default(primary_hue="orange", secondary_hue="yellow")) as demo:
    gr.Markdown("# FastVLM Inference Engine")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Settings")
            model_dropdown = gr.Dropdown(
                choices=list(MODELS.keys()),
                label="Model",
                value="Stage 3 (0.5B)"
            )
            load_btn = gr.Button("🔄 Load Model", variant="primary")
            load_status = gr.Textbox(label="Status", interactive=False, value="No model loaded")

            gr.Markdown("### 🎛️ Generation")
            temperature = gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Temperature (0=deterministic)")
            top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.1, label="Top P")

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("💬 Chat"):
                    image_input = gr.Image(type="pil", label="Upload Image")
                    chatbot = gr.Chatbot(label="Conversation", height=300, type="tuples")
                    msg = gr.Textbox(label="Your Question", placeholder="Ask about the image...")
                    with gr.Row():
                        clear = gr.Button("🗑️ Clear")
                        submit_btn = gr.Button("Send", variant="primary")
                    metrics_display = gr.Markdown("**TTFT:** - | **Speed:** - | **Tokens:** -")

                with gr.Tab("📹 Live Video"):
                    gr.Markdown("### Real-time Video Analysis")

                    with gr.Row():
                        with gr.Column(scale=2):
                            live_image_input = gr.Image(
                                sources=["webcam"],
                                streaming=True,
                                type="pil",
                                label="Camera Feed"
                            )
                        with gr.Column(scale=1):
                            preset_dropdown = gr.Dropdown(
                                choices=list(PRESET_PROMPTS.keys()),
                                label="Quick Prompts",
                                value="Describe"
                            )
                            live_prompt = gr.Textbox(
                                label="Prompt",
                                value="Describe what you see briefly.",
                                lines=2
                            )
                            frame_skip = gr.Slider(
                                1, 10, value=3, step=1,
                                label="Frame Skip (higher = smoother but slower updates)"
                            )

                    live_output = gr.Textbox(label="📝 Output", lines=2, max_lines=3)

                    with gr.Row():
                        live_status = gr.Textbox(label="Status", value="⏸️ Waiting...", interactive=False, scale=1)
                        reset_btn = gr.Button("🔄 Reset Stats", scale=1)

                    performance_display = gr.Markdown("**Stats:** No data yet")

    # Event handlers
    load_btn.click(load_model_fn, inputs=[model_dropdown], outputs=[load_status])

    preset_dropdown.change(update_prompt_from_preset, inputs=[preset_dropdown], outputs=[live_prompt])
    reset_btn.click(reset_performance, outputs=[performance_display])

    # Live Video streaming
    live_image_input.stream(
        live_inference,
        inputs=[live_image_input, live_prompt, temperature, top_p, frame_skip],
        outputs=[live_output, live_status, performance_display],
        show_progress="hidden"
    )

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history, image, temperature, top_p):
        if not history:
            return history, ""
        user_message = history[-1][0]
        for partial_response, metrics in chat(user_message, history, image, temperature, top_p):
            history[-1][1] = partial_response
            yield history, metrics

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [chatbot, image_input, temperature, top_p], [chatbot, metrics_display]
    )
    submit_btn.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [chatbot, image_input, temperature, top_p], [chatbot, metrics_display]
    )
    clear.click(lambda: (None, []), outputs=[image_input, chatbot], queue=False)

if __name__ == "__main__":
    demo.queue().launch(share=False)
