import sys
import os
import time
import threading
import torch
import gradio as gr
from PIL import Image

# Add parent directory to path to import llava
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from transformers import TextIteratorStreamer

# Global variables to hold model components
tokenizer = None
model = None
image_processor = None
context_len = None
current_model_name = None

MODELS = {
    "Stage 2 (0.5B)": "../checkpoints/llava-fastvithd_0.5b_stage2",
    "Stage 3 (0.5B)": "../checkpoints/llava-fastvithd_0.5b_stage3"
}

def load_model_fn(model_choice):
    global tokenizer, model, image_processor, context_len, current_model_name

    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), MODELS[model_choice]))

    if current_model_name == model_choice:
        return f"Model {model_choice} already loaded."

    print(f"Loading {model_choice} from {model_path}...")
    try:
        # Unload previous model if exists to save VRAM
        if model is not None:
            del model
            torch.cuda.empty_cache()

        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_name,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Set the pad token id for generation
        if tokenizer.pad_token_id is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id

        current_model_name = model_choice
        return f"Successfully loaded {model_choice}"
    except Exception as e:
        return f"Error loading model: {str(e)}"

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
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=512,
        streamer=streamer,
        use_cache=True
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

        metrics = f"**TTFT:** {ttft:.2f} ms | **TPS:** {tps:.2f} tokens/s"

        yield generated_text, metrics

    thread.join()

# UI Construction
with gr.Blocks(title="FastVLM Windows Inference") as demo:
    gr.Markdown("# FastVLM Windows Inference Engine")

    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(choices=list(MODELS.keys()), label="Select Model", value="Stage 2 (0.5B)")
            load_btn = gr.Button("Load Model")
            load_status = gr.Textbox(label="Status", interactive=False)

            image_input = gr.Image(type="pil", label="Upload Image")

            temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, label="Temperature")
            top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, label="Top P")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat", type="tuples")
            msg = gr.Textbox(label="Message")
            clear = gr.Button("Clear")
            metrics_display = gr.Markdown("**TTFT:** - | **TPS:** -")

    # Event handlers
    load_btn.click(load_model_fn, inputs=[model_dropdown], outputs=[load_status])

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history, image, temperature, top_p):
        if not history:
            return history, ""

        user_message = history[-1][0]

        # Call chat generator
        for partial_response, metrics in chat(user_message, history, image, temperature, top_p):
            history[-1][1] = partial_response
            yield history, metrics

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [chatbot, image_input, temperature, top_p], [chatbot, metrics_display]
    )

    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.queue().launch()
