#!/bin/bash
# FastVLM Video Fine-tuning Script for Exercise Form Analysis
# Uses the Qwen-based training pipeline with video support

set -e

# Run setup verification first
echo "Running setup verification..."
echo ""
bash scripts/verify_setup.sh

echo ""
read -p "Continue with training? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

echo ""

# Configuration
MODEL_PATH="checkpoints/llava-fastvithd_0.5b_stage3"
DATA_PATH="dataset/fastvlm_train.json"
EVAL_DATA_PATH="dataset/fastvlm_val.json"
IMAGE_FOLDER="dataset"
OUTPUT_DIR="models/exercise-video-finetuned"
NUM_VIDEO_FRAMES=4  # Reduced for memory efficiency

# Training hyperparameters
BATCH_SIZE=1
GRAD_ACCUM_STEPS=8
LEARNING_RATE=2e-5
NUM_EPOCHS=3
MODEL_MAX_LENGTH=2048
WARMUP_RATIO=0.03

# WandB configuration
export WANDB_PROJECT="fastvlm"
export WANDB_ENTITY="fyp-21"
export WANDB_NAME="fastvlm-finetune-${TIMESTAMP}"
export WANDB_LOG_MODEL="end"  # Log model at end of training (options: false, end, checkpoint)

# Evaluation configuration
EVAL_STRATEGY="steps"
EVAL_STEPS=50
SAVE_STEPS=50

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="models/training_log_${TIMESTAMP}.log"
HPARAMS_FILE="models/hyperparameters_${TIMESTAMP}.json"

# Save hyperparameters to JSON
cat > "$HPARAMS_FILE" << EOF
{
  "timestamp": "$TIMESTAMP",
  "model_path": "$MODEL_PATH",
  "data_path": "$DATA_PATH",
  "eval_data_path": "$EVAL_DATA_PATH",
  "image_folder": "$IMAGE_FOLDER",
  "output_dir": "$OUTPUT_DIR",
  "num_video_frames": $NUM_VIDEO_FRAMES,
  "batch_size": $BATCH_SIZE,
  "gradient_accumulation_steps": $GRAD_ACCUM_STEPS,
  "effective_batch_size": $((BATCH_SIZE * GRAD_ACCUM_STEPS)),
  "learning_rate": "$LEARNING_RATE",
  "num_epochs": $NUM_EPOCHS,
  "model_max_length": $MODEL_MAX_LENGTH,
  "vision_tower": "mobileclip_l_1024",
  "mm_projector_type": "mlp2x_gelu",
  "mm_vision_select_layer": -2,
  "image_aspect_ratio": "pad",
  "bf16": true,
  "tf32": true,
  "gradient_checkpointing": true,
  "weight_decay": 0.0,
  "warmup_ratio": $WARMUP_RATIO,
  "lr_scheduler_type": "cosine",
  "dataloader_num_workers": 2,
  "evaluation_strategy": "$EVAL_STRATEGY",
  "eval_steps": $EVAL_STEPS,
  "save_steps": $SAVE_STEPS,
  "wandb_project": "$WANDB_PROJECT",
  "wandb_entity": "$WANDB_ENTITY",
  "wandb_run_name": "fastvlm-finetune-${TIMESTAMP}"
}
EOF

echo "=============================================="
echo "FastVLM Video Fine-tuning for Exercise Analysis"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Eval Data: $EVAL_DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "Video frames: $NUM_VIDEO_FRAMES"
echo "Batch size: $BATCH_SIZE"
echo "Gradient accumulation: $GRAD_ACCUM_STEPS"
echo "Effective batch size: $((BATCH_SIZE * GRAD_ACCUM_STEPS))"
echo "Learning rate: $LEARNING_RATE"
echo "Epochs: $NUM_EPOCHS"
echo "Evaluation: $EVAL_STRATEGY every $EVAL_STEPS steps"
echo "Saving: Every $SAVE_STEPS steps"
echo "WandB: $WANDB_ENTITY/$WANDB_PROJECT ($WANDB_NAME)"
echo "Log file: $LOG_FILE"
echo "Hyperparameters: $HPARAMS_FILE"
echo "=============================================="

# Set environment variables for better logging
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=info

# Run training with output to both console and log file
python -u llava/train/train_mem.py \
    --model_name_or_path "$MODEL_PATH" \
    --version qwen_v2 \
    --data_path "$DATA_PATH" \
    --validation_data_path "$EVAL_DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --vision_tower mobileclip_l_1024 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --num_video_frames $NUM_VIDEO_FRAMES \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --run_name "fastvlm-finetune-${TIMESTAMP}" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --evaluation_strategy "$EVAL_STRATEGY" \
    --eval_steps $EVAL_STEPS \
    --save_strategy "steps" \
    --save_steps $SAVE_STEPS \
    --save_total_limit 2 \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0.0 \
    --warmup_ratio $WARMUP_RATIO \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to wandb 2>&1 | tee "$LOG_FILE"

echo "=============================================="
echo "Training complete!"
echo "Model saved to: $OUTPUT_DIR"
echo "Log saved to: $LOG_FILE"
echo "Hyperparameters saved to: $HPARAMS_FILE"
echo "=============================================="

# Generate training plots
echo ""
echo "Generating training plots..."
python utils/plot_training_stats.py --log_file "$LOG_FILE" --output_dir "plots/fastvlm_exercise"

echo ""
echo "=============================================="
echo "Inference Examples"
echo "=============================================="
echo ""
echo "To run inference on a video:"
echo "  python utils/inference_single.py --model-path $OUTPUT_DIR --input-file videos/00000340.mp4"
echo ""
echo "To run inference on an image:"
echo "  python utils/inference_single.py --model-path $OUTPUT_DIR --input-file path/to/image.jpg"
echo ""
echo "Custom prompt example:"
echo "  python utils/inference_single.py --model-path $OUTPUT_DIR --input-file videos/00000340.mp4 --prompt \"What exercise is being performed?\""
echo "=============================================="

# Prompt for HuggingFace upload
echo ""
echo "=============================================="
echo "HuggingFace Hub Upload"
echo "=============================================="

# Generate HF repo name with same timestamp
HF_REPO_NAME="fastvlm-finetune-${TIMESTAMP}"

read -p "Would you like to upload the model to HuggingFace? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    read -p "Make repository private? (y/n) " -n 1 -r PRIVATE_REPLY
    echo ""

    PRIVATE_FLAG=""
    if [[ $PRIVATE_REPLY =~ ^[Yy]$ ]]; then
        PRIVATE_FLAG="--private"
    fi

    echo "Uploading to HuggingFace Hub..."
    echo "Repository: EdgeVLM-Labs/${HF_REPO_NAME}"
    python utils/hf_upload.py \
        --model_path "$OUTPUT_DIR" \
        --repo_name "$HF_REPO_NAME" \
        --org "EdgeVLM-Labs" \
        $PRIVATE_FLAG

    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Model uploaded successfully!"
    else
        echo ""
        echo "⚠ Upload failed. You can try again later with:"
        echo "  python utils/hf_upload.py --model_path $OUTPUT_DIR --repo_name $HF_REPO_NAME --org EdgeVLM-Labs"
    fi
else
    echo ""
    echo "Skipping HuggingFace upload."
    echo "To upload later, run:"
    echo "  python utils/hf_upload.py --model_path $OUTPUT_DIR --repo_name $HF_REPO_NAME --org EdgeVLM-Labs"
fi

echo ""
echo "=============================================="
echo "✅ All done!"
echo "=============================================="