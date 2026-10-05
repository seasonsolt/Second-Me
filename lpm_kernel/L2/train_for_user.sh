#!/bin/bash

# Exit when training fails; the Web workflow must not advance to merge.
set -euo pipefail

# Initialize variables
LEARNING_RATE="2e-4"
NUM_TRAIN_EPOCHS="3"
CONCURRENCY_THREADS="2"
DATA_SYNTHESIS_MODE="low"
HALF=False
USE_CUDA=False  # Default to False, will be overridden by parameter
IS_COT=False
# Optional overrides from the Web settings; defaults keep the legacy batch settings.
# 4096 tokens fits the 4096-character reference-memory budget in training prompts.
BATCH_SIZE="2"
GRADIENT_ACCUMULATION_STEPS=""  # Empty means use CONCURRENCY_THREADS
MAX_LENGTH="4096"
GRADIENT_CHECKPOINTING=True

# Process parameters
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --lr) LEARNING_RATE="$2"; shift ;;
        --epochs) NUM_TRAIN_EPOCHS="$2"; shift ;;
        --threads) CONCURRENCY_THREADS="$2"; shift ;;
        --mode) DATA_SYNTHESIS_MODE="$2"; shift ;;
        --cuda) 
            # Convert string to lowercase for consistent comparison
            cuda_value=$(echo "$2" | tr '[:upper:]' '[:lower:]')
            if [[ "$cuda_value" == "true" || "$cuda_value" == "1" || "$cuda_value" == "yes" ]]; then
                USE_CUDA=True
                echo "CUDA enabled by user configuration."
            else
                USE_CUDA=False
                echo "CUDA disabled by user configuration."
            fi
            shift ;;
        --is_cot) IS_COT="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --grad_accum) GRADIENT_ACCUMULATION_STEPS="$2"; shift ;;
        --max_length) MAX_LENGTH="$2"; shift ;;
        --grad_checkpoint) GRADIENT_CHECKPOINTING="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-$CONCURRENCY_THREADS}"

# Explicitly log the CUDA setting passed from the command line
echo "CUDA parameter received: $USE_CUDA"

# Verify CUDA availability if enabled
if [[ "$USE_CUDA" == "True" ]]; then
    # Set CUDA environment variables to ensure PyTorch detects GPU
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    
    # Set CUDA_LAUNCH_BLOCKING to 0 for async operations (better performance)
    export CUDA_LAUNCH_BLOCKING=0
    echo "CUDA_LAUNCH_BLOCKING set to 0 for better performance"
else
    # Explicitly disable CUDA
    export CUDA_VISIBLE_DEVICES=""
    echo "CUDA_VISIBLE_DEVICES explicitly disabled"
fi

# Log the parameters being used
echo "Using training parameters:"
echo "  Learning rate: $LEARNING_RATE"
echo "  Number of epochs: $NUM_TRAIN_EPOCHS"
echo "  Concurrency threads: $CONCURRENCY_THREADS"
echo "  Data synthesis mode: $DATA_SYNTHESIS_MODE"
echo "  Use CUDA: $USE_CUDA"
echo "  Is chain of thought: $IS_COT"
echo "  Batch size: $BATCH_SIZE"
echo "  Gradient accumulation steps: $GRADIENT_ACCUMULATION_STEPS"
echo "  Max sequence length: $MAX_LENGTH"
echo "  Gradient checkpointing: $GRADIENT_CHECKPOINTING"

# If concurrency threads are set, configure related environment variables
if [ "$CONCURRENCY_THREADS" != "1" ]; then
  # Limit the number of parallel threads to avoid memory issues
  export OMP_NUM_THREADS=$CONCURRENCY_THREADS
  export MKL_NUM_THREADS=$CONCURRENCY_THREADS
  export NUMEXPR_NUM_THREADS=$CONCURRENCY_THREADS
  # Add torch-specific threading controls
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
  echo "Set thread environment variables to $CONCURRENCY_THREADS"
fi

# Add BF16 option based on the platform and CUDA availability
if [ "${PLATFORM:-}" != "apple" ] && [ "$USE_CUDA" == "True" ]; then
  HALF=True
  echo "Enabling BF16 half precision for non-Apple platform with CUDA"
else
  echo "Using standard precision (not using BF16)"
fi

# Print environment for debugging
echo "Environment configuration:"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF:-}"
echo "  Using half precision: ${HALF}"

# Execute training script with parameters from environment variables
"${PYTHON_EXECUTABLE:-python}" lpm_kernel/L2/train.py \
  --seed 42 \
  --model_name_or_path "${MODEL_BASE_PATH}" \
  --user_name "${USER_NAME}" \
  --dataset_name "resources/L2/data/merged.json" \
  --chat_template_format "none" \
  --add_special_tokens False \
  --append_concat_token False \
  --max_length "$MAX_LENGTH" \
  --num_train_epochs $NUM_TRAIN_EPOCHS \
  --save_total_limit 2 \
  --logging_steps 20 \
  --log_level "info" \
  --logging_strategy "steps" \
  --save_strategy "steps" \
  --save_steps 5 \
  --push_to_hub False \
  --bf16 $HALF \
  --packing False \
  --learning_rate $LEARNING_RATE \
  --lr_scheduler_type "cosine" \
  --weight_decay 1e-4 \
  --max_grad_norm 0.3 \
  --output_dir "${MODEL_PERSONAL_DIR}" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
  --use_reentrant False \
  --use_peft_lora True \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --lora_target_modules "all-linear" \
  --use_4bit_quantization False \
  --use_nested_quant False \
  --bnb_4bit_compute_dtype "bfloat16" \
  --is_cot $IS_COT \
  --use_cuda $USE_CUDA

