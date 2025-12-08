#!/bin/bash

# FastVLM Test Inference and Evaluation Report Generator
# This script runs inference on the test set and generates an evaluation report

set -e  # Exit on error

echo "========================================="
echo "FastVLM Test Inference & Evaluation"
echo "========================================="

# Default values
MODEL_PATH=""
HF_REPO=""
MODEL_BASE=""
TEST_JSON="dataset/fastvlm_test.json"
DATA_PATH="dataset"
OUTPUT_DIR=""
DEVICE="cuda"
MAX_NEW_TOKENS=64
NUM_FRAMES=4
CONV_MODE="qwen_2"
LIMIT=""
NO_BERT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --hf_repo)
            HF_REPO="$2"
            shift 2
            ;;
        --model_base)
            MODEL_BASE="$2"
            shift 2
            ;;
        --test_json)
            TEST_JSON="$2"
            shift 2
            ;;
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --max_new_tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --num_frames)
            NUM_FRAMES="$2"
            shift 2
            ;;
        --conv_mode)
            CONV_MODE="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --no-bert)
            NO_BERT="--no-bert"
            shift
            ;;
        -h|--help)
            echo "Usage: bash scripts/run_inference.sh [--model_path <path> | --hf_repo <repo>] [options]"
            echo ""
            echo "Model Source (one required):"
            echo "  --model_path      Path to local finetuned FastVLM model directory"
            echo "  --hf_repo         HuggingFace repository URL/ID (e.g., EdgeVLM-Labs/fastvlm-exercise-20241128)"
            echo ""
            echo "Optional:"
            echo "  --model_base      Base model path for LoRA adapters"
            echo "  --test_json       Path to test set JSON (default: dataset/fastvlm_test.json)"
            echo "  --data_path       Base path for video files (default: dataset)"
            echo "  --output_dir      Output directory for results (default: model directory)"
            echo "  --device          Device to use: cuda/mps/cpu (default: cuda)"
            echo "  --max_new_tokens  Max tokens to generate (default: 64)"
            echo "  --num_frames      Number of frames to extract per video (default: 4)"
            echo "  --conv_mode       Conversation template (default: qwen_2)"
            echo "  --limit           Limit number of samples (for testing)"
            echo "  --no-bert         Skip BERT similarity (faster evaluation)"
            echo ""
            echo "Examples:"
            echo "  # Using local finetuned model:"
            echo "  bash scripts/run_inference.sh --model_path models/exercise-video-finetuned"
            echo ""
            echo "  # Using HuggingFace model:"
            echo "  bash scripts/run_inference.sh --hf_repo EdgeVLM-Labs/fastvlm-exercise-20241128"
            echo ""
            echo "  # With options:"
            echo "  bash scripts/run_inference.sh --model_path models/exercise-video-finetuned --limit 10 --device cuda"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate: either model_path or hf_repo must be provided
if [ -z "$MODEL_PATH" ] && [ -z "$HF_REPO" ]; then
    echo "❌ Error: Either --model_path or --hf_repo is required"
    echo "Use --help for usage information"
    exit 1
fi

# If HF repo is provided, use it as model path
if [ -n "$HF_REPO" ]; then
    # Extract repo ID from URL if full URL is provided
    # e.g., https://huggingface.co/EdgeVLM-Labs/fastvlm-exercise -> EdgeVLM-Labs/fastvlm-exercise
    if [[ "$HF_REPO" == *"huggingface.co/"* ]]; then
        HF_REPO=$(echo "$HF_REPO" | sed 's|.*huggingface.co/||' | sed 's|/$||')
    fi

    echo "🤗 Using HuggingFace model: $HF_REPO"
    MODEL_PATH="$HF_REPO"

    # Set default output directory for HF models
    if [ -z "$OUTPUT_DIR" ]; then
        # Create output dir based on repo name
        REPO_NAME=$(echo "$HF_REPO" | sed 's|/|_|g')
        OUTPUT_DIR="results/hf_inference_${REPO_NAME}"
        mkdir -p "$OUTPUT_DIR"
    fi
else
    # Validate local model path exists
    if [ ! -e "$MODEL_PATH" ]; then
        echo "❌ Error: Model path not found: $MODEL_PATH"
        exit 1
    fi
fi

if [ ! -f "$TEST_JSON" ]; then
    echo "❌ Error: Test JSON not found: $TEST_JSON"
    exit 1
fi

# Set output directory if not already set
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$MODEL_PATH"
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR" 2>/dev/null || true

PREDICTIONS_FILE="${OUTPUT_DIR}/test_predictions.json"
REPORT_FILE="${OUTPUT_DIR}/test_evaluation_report.xlsx"

echo ""
echo "Configuration:"
echo "  Model path:      $MODEL_PATH"
if [ -n "$MODEL_BASE" ]; then
    echo "  Model base:      $MODEL_BASE"
fi
echo "  Test JSON:       $TEST_JSON"
echo "  Data path:       $DATA_PATH"
echo "  Output dir:      $OUTPUT_DIR"
echo "  Device:          $DEVICE"
echo "  Max new tokens:  $MAX_NEW_TOKENS"
echo "  Num frames:      $NUM_FRAMES"
echo "  Conv mode:       $CONV_MODE"
if [ -n "$LIMIT" ]; then
    echo "  Sample limit:    $LIMIT"
fi
echo "========================================="
echo ""

# Step 1: Run inference
echo "[Step 1/2] Running inference on test set..."
echo "========================================="

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

MODEL_BASE_ARG=""
if [ -n "$MODEL_BASE" ]; then
    MODEL_BASE_ARG="--model_base $MODEL_BASE"
fi

python utils/test_inference.py \
    --model_path "$MODEL_PATH" \
    --test_json "$TEST_JSON" \
    --data_path "$DATA_PATH" \
    --output "$PREDICTIONS_FILE" \
    --device "$DEVICE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --num_frames "$NUM_FRAMES" \
    --conv_mode "$CONV_MODE" \
    $MODEL_BASE_ARG \
    $LIMIT_ARG

if [ $? -ne 0 ]; then
    echo "❌ Inference failed!"
    exit 1
fi

echo ""
echo "✓ Inference complete! Predictions saved to: $PREDICTIONS_FILE"
echo ""

# Step 2: Generate evaluation report
echo "[Step 2/2] Generating evaluation report..."
echo "========================================="

python utils/generate_test_report.py \
    --predictions "$PREDICTIONS_FILE" \
    --output "$REPORT_FILE" \
    $NO_BERT

if [ $? -ne 0 ]; then
    echo "⚠ Warning: Failed to generate evaluation report"
    echo "  You can generate it later with:"
    echo "  python utils/generate_test_report.py --predictions $PREDICTIONS_FILE"
else
    echo ""
    echo "✓ Evaluation report saved to: $REPORT_FILE"
fi

echo ""
echo "========================================="
echo "✅ All steps complete!"
echo "========================================="
echo ""
echo "Output files:"
echo "  Predictions: $PREDICTIONS_FILE"
echo "  Report:      $REPORT_FILE"
echo ""
echo "========================================="
