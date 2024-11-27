# RWKV-SFT

This is a simple RWKV SFT code repository.

## Installation

To use this code, you need to have the following dependencies installed:

- python 3.10
- torch 2.0.0+
- cuda 12.1+
- pytorch-lightning==1.9.5

## Usage

1. Clone this repository.
2. Run the `train.py` file.
3. see `scripts/train/rwkv1b6_sft` folder for more details.

## comandline arguments

- `--load_model`: Load pre-trained model from the given path.
- `--data_file`: Path to the data file, which is a llama conversation json file.
- `--epoch_steps`: Number of steps per epoch, which is a window for logging.
- `--epoch_count`: Number of epochs to train. epoch_count x epoch_steps = total steps. micro_bsz x total steps = total samples.
- `--micro_bsz`: Micro batch size. mirco_bsz*accumulate_grad_batches = real batch size.
- `--num_layers_to_freeze`: Number of layers to freeze, starting from the first layer.

```bash
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
```

## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
