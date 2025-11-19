#!/usr/bin/env python3
"""
Example script demonstrating how to use FastVLM's video fine-tuning capabilities.

This script shows how to:
1. Prepare a video dataset
2. Configure training for video fine-tuning
3. Train a model on video data
"""

import json
import os

def create_example_dataset():
    """
    Create an example training dataset JSON file for video fine-tuning.
    
    This demonstrates the correct format for a mixed image/video dataset.
    """
    
    dataset = [
        # Example 1: Video description
        {
            "image": "videos/cooking_demo.mp4",
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nDescribe step by step what is happening in this cooking video."
                },
                {
                    "from": "gpt",
                    "value": "In this cooking demonstration: 1) First, the chef preheats the pan. 2) Next, they add olive oil. 3) Then they add chopped vegetables. 4) The ingredients are sautéed until golden brown. 5) Finally, seasonings are added and mixed well."
                }
            ]
        },
        
        # Example 2: Action recognition
        {
            "image": "videos/sports_clip.mp4",
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nWhat sport is being played and what specific action is shown?"
                },
                {
                    "from": "gpt",
                    "value": "The video shows basketball. Specifically, a player is performing a jump shot. They dribble towards the basket, jump, and release the ball with proper shooting form towards the hoop."
                }
            ]
        },
        
        # Example 3: Video QA
        {
            "image": "videos/nature_scene.mp4",
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nHow many different animals can you see in this video?"
                },
                {
                    "from": "gpt",
                    "value": "I can see three different types of animals in this video: two deer grazing in the meadow, a red fox walking along the edge of the forest, and several birds flying overhead."
                }
            ]
        },
        
        # Example 4: Mixed with static image
        {
            "image": "images/diagram.jpg",
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nExplain this diagram."
                },
                {
                    "from": "gpt",
                    "value": "This diagram illustrates the water cycle, showing evaporation, condensation, precipitation, and collection processes."
                }
            ]
        },
        
        # Example 5: Video summarization
        {
            "image": "videos/tutorial.mp4",
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nProvide a brief summary of this tutorial video."
                },
                {
                    "from": "gpt",
                    "value": "This tutorial demonstrates how to change a bicycle tire. The process includes: removing the wheel, deflating the old tire, using tire levers to remove it, installing the new tire, inflating it to proper pressure, and reattaching the wheel."
                }
            ]
        }
    ]
    
    # Save to file
    output_path = "example_video_dataset.json"
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"✓ Created example dataset: {output_path}")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Video samples: {sum(1 for item in dataset if item['image'].endswith('.mp4'))}")
    print(f"  Image samples: {sum(1 for item in dataset if not item['image'].endswith('.mp4'))}")
    
    return output_path

def create_training_script():
    """
    Create an example training script with recommended parameters.
    """
    
    script_content = """#!/bin/bash
# FastVLM Video Fine-tuning Example Script

# Configuration
MODEL_PATH="path/to/base/fastvlm/model"
DATA_PATH="example_video_dataset.json"
IMAGE_FOLDER="path/to/video/and/image/folder"
OUTPUT_DIR="./checkpoints/fastvlm-video-finetuned"

# Training parameters
NUM_FRAMES=8  # Number of frames to extract per video
BATCH_SIZE=4
GRAD_ACCUM=1
EPOCHS=3
LEARNING_RATE=2e-3
MAX_LENGTH=2048

# Run training
python llava/train/train.py \\
    --model_name_or_path $MODEL_PATH \\
    --version v1 \\
    --data_path $DATA_PATH \\
    --image_folder $IMAGE_FOLDER \\
    --vision_tower timm/fastvit_sa12.apple_dist_in1k \\
    --mm_projector_type mlp2x_gelu \\
    --mm_vision_select_layer -2 \\
    --mm_use_im_start_end False \\
    --mm_use_im_patch_token False \\
    --image_aspect_ratio pad \\
    --num_video_frames $NUM_FRAMES \\
    --bf16 True \\
    --output_dir $OUTPUT_DIR \\
    --num_train_epochs $EPOCHS \\
    --per_device_train_batch_size $BATCH_SIZE \\
    --per_device_eval_batch_size $BATCH_SIZE \\
    --gradient_accumulation_steps $GRAD_ACCUM \\
    --evaluation_strategy "no" \\
    --save_strategy "steps" \\
    --save_steps 500 \\
    --save_total_limit 2 \\
    --learning_rate $LEARNING_RATE \\
    --weight_decay 0. \\
    --warmup_ratio 0.03 \\
    --lr_scheduler_type "cosine" \\
    --logging_steps 1 \\
    --tf32 True \\
    --model_max_length $MAX_LENGTH \\
    --gradient_checkpointing True \\
    --dataloader_num_workers 4 \\
    --lazy_preprocess True \\
    --report_to wandb

echo "Training completed! Model saved to: $OUTPUT_DIR"
"""
    
    script_path = "train_video_example.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)  # Make executable
    
    print(f"✓ Created training script: {script_path}")
    print("  Edit the paths in the script before running")
    
    return script_path

def print_usage_guide():
    """
    Print a quick usage guide.
    """
    
    guide = """
╔════════════════════════════════════════════════════════════════════╗
║          FastVLM Video Fine-tuning Quick Start Guide              ║
╚════════════════════════════════════════════════════════════════════╝

1. INSTALLATION
   ────────────
   pip install decord

2. DATASET PREPARATION
   ───────────────────
   • Place videos in a folder (e.g., ./data/videos/)
   • Create a JSON file with format shown in example_video_dataset.json
   • Videos are automatically detected by file extension

3. SUPPORTED FORMATS
   ─────────────────
   .mp4, .avi, .mov, .mkv, .flv, .wmv, .webm, .m4v

4. KEY PARAMETERS
   ──────────────
   --num_video_frames 8      # Frames to extract (4-16 recommended)
   --model_max_length 2048   # Context length
   --image_aspect_ratio pad  # Preprocessing mode

5. MEMORY OPTIMIZATION
   ───────────────────
   • Fewer frames = less memory
   • FastVLM uses ~100 tokens/frame (vs 576 for CLIP)
   • 8 frames ≈ 800 tokens, leaving room for ~1200 tokens of text

6. TRAINING
   ────────
   bash train_video_example.sh

7. DOCUMENTATION
   ─────────────
   See docs/VIDEO_FINETUNING.md for detailed information

8. TOKEN BUDGET EXAMPLE
   ────────────────────
   8 frames × 100 tokens/frame = 800 visual tokens
   2048 total tokens - 800 visual = 1248 tokens for text
   
   vs. Standard CLIP:
   8 frames × 576 tokens/frame = 4608 visual tokens (exceeds 4096!)

╚════════════════════════════════════════════════════════════════════╝
"""
    
    print(guide)

def main():
    """
    Main function to set up example files and display usage information.
    """
    
    print("="*70)
    print("FastVLM Video Fine-tuning Setup")
    print("="*70)
    print()
    
    # Create example files
    dataset_path = create_example_dataset()
    print()
    
    script_path = create_training_script()
    print()
    
    # Print usage guide
    print_usage_guide()
    
    print("\n" + "="*70)
    print("Setup complete! Next steps:")
    print("="*70)
    print("1. Edit train_video_example.sh with your actual paths")
    print("2. Ensure you have video files in the specified folder")
    print("3. Install decord: pip install decord")
    print("4. Run: bash train_video_example.sh")
    print("="*70)

if __name__ == "__main__":
    main()
