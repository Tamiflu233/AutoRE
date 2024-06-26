#!/bin/bash
source /root/anaconda3/bin/activate autore
# 定义模式变量
mode="D_R_H_F_desc"

# 定义任务列表
declare -A tasks
tasks["mistral"]="dataset=train_$mode eval_path=test_$mode cache_path=autore/mistral/$mode/train eval_cache_path=autore/mistral/$mode/test output_dir_base=ckpt/mistral/$mode model_name_or_path=checkpoints/Mistral-7B-Instruct-v0.2 template=mistral learning_rate=5e-5 num_train_epochs=6.0"

for task_name in "${!tasks[@]}"; do
  task_config=${tasks[$task_name]}

  # 解析配置字符串
  for kv in $task_config; do
    key=${kv%%=*}
    value=${kv#*=}
    declare $key=$value
  done

  # 设置 output_dir，包含学习率信息
  output_dir="${output_dir_base}_lr${learning_rate}_gpu"
  export WANDB_PROJECT_NAME="${task_name}_${mode}_${learning_rate}"

  log_dir="$output_dir"
  parent_dir=$(dirname "$output_dir")
  log_dir="$parent_dir/log"
  if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
  fi

  # 执行训练命令
#  deepspeed --num_gpus 8 --master_port=9901 src/train_bash.py \
#    --deepspeed ds_config/stage2.json \
  CUDA_VISIBLE_DEVICES=0 /root/anaconda3/envs/autore/bin/python src/train_bash.py \
    --stage sft \
    --do_train \
    --evaluation_strategy "steps" \
    --model_name_or_path "$model_name_or_path" \
    --dataset "$dataset" \
    --eval_path "$eval_path" \
    --cache_path "$cache_path" \
    --eval_cache_path "$eval_cache_path" \
    --template "$template" \
    --output_dir "$output_dir" \
    --finetuning_type lora \
    --cutoff_len 2048 \
    --lora_target q_proj,v_proj \
    --save_total_limit 5 \
    --lora_r 300 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --quantization_bit 4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_strategy "steps" \
    --save_steps 100 \
    --eval_steps 100 \
    --learning_rate "$learning_rate" \
    --num_train_epochs "$num_train_epochs" \
    --plot_loss \
    --fp16 2>&1 | tee -a "$log_dir/${task_name}_${mode}_${learning_rate}.log"
done