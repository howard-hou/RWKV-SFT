export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline

# 切换到脚本所在目录的上两级目录
cd "$(dirname "$(dirname "$0")")/.."

# 打印当前工作目录
echo "Current working directory: $(pwd)"


python train.py --load_model RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth \
    --wandb "rwkv1b6-sft" --proj_dir out/rwkv1b6-sft \
    --data_file path_to_json \
    --data_type "json" --vocab_size 65536 \
    --ctx_len 2048 --epoch_steps 1000 --epoch_count 1 --epoch_begin 0 --epoch_save 0 \
    --micro_bsz 4 --accumulate_grad_batches 16 --n_layer 24 --n_embd 2048 --pre_ffn 0 \
    --lr_init 6e-5 --lr_final 1.5e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 1 \
    --num_layers_to_freeze 12 \
    --enable_progress_bar True
