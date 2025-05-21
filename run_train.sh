# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Get into python virtual environment
source venv/bin/activate

# File path with full bf16 safetensors
INPUT_MODEL_FILENAME="/mnt/astrodata/llm/models/meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_MODEL_FILENAME="/mnt/astrodata/llm/models/ubergarm/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-ParetoQ-2bpw"

# Specify 2 bpw Training Mode
W_BITS=2

# Defaults to 2 but I keep ooming on 1B model with 24GB VRAM lmao... wtf...
NUM_BATCHES=1

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nnodes=1 --nproc_per_node=1 train.py \
--local_dir "./" \
--output_dir "./checkpoints/" \
--input_model_filename "$INPUT_MODEL_FILENAME" \
--output_model_filename "$OUTPUT_MODEL_FILENAME" \
--train_data_local_path "./chunk1.jsonl" \
--do_train True \
--do_eval False \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--logging_dir "./checkpoints/runs/current/" \
--num_train_epochs 1 \
--per_device_train_batch_size "$NUM_BATCHES" \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 500 \
--report_to "tensorboard" \
--save_total_limit 2 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0. \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 False \
--gradient_checkpointing False \
--qat True \
--w_bits "$W_BITS"
