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
is_generating_live = False
last_live_output = ""

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

def live_inference(image, prompt, temperature, top_p):
    global is_generating_live, last_live_output, model, tokenizer, image_processor

    if image is None:
        return last_live_output, "⏸️ Waiting for webcam..."

    if model is None:
        return "⚠️ Model not loaded. Please load a model in the Chat tab.", "❌ Model not loaded"

    # If busy, skip this frame and return previous result
    if is_generating_live:
        return last_live_output, "⏳ Processing previous frame..."

    is_generating_live = True

    try:
        # Prepare prompt
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

        # Tokenize
        start_time = time.time()
        input_ids = tokenizer_image_token(prompt_str, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

        # Process Image
        image_tensor = process_images([image], image_processor, model.config)[0]

        # Generate with stricter parameters for live video
        with torch.inference_mode():
            output_ids = model.generate(
                inputs=input_ids,
                images=image_tensor.unsqueeze(0).half(),
                image_sizes=[image.size],
                do_sample=False,  # Greedy decoding for consistency
                max_new_tokens=30,  # Reduced for faster, cleaner output
                use_cache=True,
                repetition_penalty=1.2,  # Penalize repetition
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode the generated tokens
        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # Clean up the output - extract only the first sentence/response
        output_text = clean_live_output(output_text)

        # Debug: print to console
        print(f"Generated text: '{output_text}' (length: {len(output_text)})")

        # Update last output even if empty (to show we processed)
        if output_text:
            last_live_output = output_text
        else:
            output_text = "[No text generated]"

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        status = f"✅ Processed in {processing_time:.0f}ms"

        return output_text, status

    except Exception as e:
        print(f"Live inference error: {e}")
        error_msg = f"Error: {str(e)[:100]}"
        return error_msg, "❌ Error occurred"
    finally:
        is_generating_live = False


def clean_live_output(text):
    """Clean up model output for live video display."""
    # Remove common repetitive patterns
    if "Answer:" in text:
        text = text.split("Answer:")[0].strip()
    
    # Remove code blocks
    if "```" in text:
        text = text.split("```")[0].strip()
    
    # Take only the first sentence if multiple exist
    for delimiter in ['\n\n', '\n']:
        if delimiter in text:
            text = text.split(delimiter)[0].strip()
            break
    
    # Limit length
    if len(text) > 150:
        text = text[:150].rsplit(' ', 1)[0] + "..."
    
    return text


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

    # Only use sampling parameters if temperature > 0
    if temperature > 0:
        generation_kwargs = dict(
            inputs=input_ids,
            images=image_tensor.unsqueeze(0).half(),
            image_sizes=[image.size],
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=256,
            streamer=streamer,
            use_cache=True
        )
    else:
        generation_kwargs = dict(
            inputs=input_ids,
            images=image_tensor.unsqueeze(0).half(),
            image_sizes=[image.size],
            do_sample=False,
            max_new_tokens=256,
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

            temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, label="Temperature")
            top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, label="Top P")

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("Chat"):
                    image_input = gr.Image(type="pil", label="Upload Image")
                    chatbot = gr.Chatbot(label="Chat", type="tuples")
                    msg = gr.Textbox(label="Message")
                    clear = gr.Button("Clear")
                    metrics_display = gr.Markdown("**TTFT:** - | **TPS:** -")

                with gr.Tab("Live Video"):
                    gr.Markdown("### Real-time Video Inference")
                    gr.Markdown("💡 **Tip:** The model will continuously analyze webcam frames. Load a model first in the Chat tab.")
                    live_image_input = gr.Image(sources=["webcam"], streaming=True, type="pil", label="Live Camera")
                    live_prompt = gr.Textbox(label="Prompt", value="Describe the image in English. Output should be brief, about 15 words or less.")
                    live_output = gr.Textbox(label="Live Output", lines=3)
                    live_status = gr.Textbox(label="Status", value="⏸️ Waiting...", interactive=False)

    # Event handlers
    load_btn.click(load_model_fn, inputs=[model_dropdown], outputs=[load_status])

    # Live Video Event - triggers on every new frame from webcam
    live_image_input.stream(
        live_inference,
        inputs=[live_image_input, live_prompt, temperature, top_p],
        outputs=[live_output, live_status],
        show_progress=False
    )

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
