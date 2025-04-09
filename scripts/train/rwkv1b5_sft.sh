export CUDA_VISIBLE_DEVICES=0,1
export WANDB_MODE=offline

# 切换到脚本所在目录的上两级目录
cd "$(dirname "$(dirname "$0")")/.."

# 打印当前工作目录
echo "Current working directory: $(pwd)"


python train.py --load_model ../RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth \
    --wandb "rwkv1b5-sft" --proj_dir out/rwkv1b5-sft \
    --data_file ../output.json \
    --data_type "json" --vocab_size 65536 \
    --ctx_len 2048 --epoch_steps 1000 --epoch_count 10 --epoch_begin 0 --epoch_save 1 \
    --micro_bsz 32 --accumulate_grad_batches 4 --n_layer 24 --n_embd 2048 --pre_ffn 0 \
    --lr_init 6e-5 --lr_final 1.5e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 2 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 1 \
    --num_layers_to_freeze 0 \
    --enable_progress_bar True
