from llava.train.train_qwen import train

if __name__ == "__main__":
    train(attn_implementation="sdpa")  # Use PyTorch's native SDPA instead of flash_attention_2
