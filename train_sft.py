"""
Poxiao SFT 训练脚本 — 支持 LoRA 微调
用法:
  torchrun --nproc_per_node=4 train_sft.py \
      --data_path sft_general.jsonl \
      --from_weight pretrain_best.pth \
      --use_lora --lora_r 8 --lora_alpha 16 \
      --epochs 1 --batch_size 32 --learning_rate 1e-5
"""
import os
import sys
import math
import time
import argparse
import warnings
import torch
import torch.nn as nn
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PoxiaoConfig
from model import PoxiaoForCausalLM
from tokenizer import PoxiaoTokenizer
from dataset import SFTDataset
from lora import apply_lora_to_model, save_lora_weights, load_lora_weights

warnings.filterwarnings("ignore")


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(msg):
    if is_main():
        print(msg, flush=True)


def get_lr(step, total_steps, max_lr, min_lr=1e-6, warmup_steps=50):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def setup_seed(seed):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if dist.is_initialized():
        torch.cuda.manual_seed(seed + dist.get_rank())
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_distributed():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=50):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i, (input_ids, labels) in enumerate(val_loader):
        if i >= max_batches:
            break
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        res = model(input_ids, labels=labels)
        n_tokens = (labels != -100).sum().item()
        total_loss += res.loss.item() * n_tokens
        total_tokens += n_tokens
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))
    model.train()
    return avg_loss, ppl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poxiao SFT Training")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="../out/sft")
    parser.add_argument("--from_weight", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    # LoRA
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    # LoRA 权重 (可选)
    parser.add_argument("--lora_weights", type=str, default=None)
    args = parser.parse_args()

    # 分布式
    local_rank = init_distributed()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    # 模型
    lm_config = PoxiaoConfig(
        hidden_size=768, num_hidden_layers=8,
        max_position_embeddings=args.max_seq_len, rope_theta=1e4,
    )
    tokenizer = PoxiaoTokenizer()
    model = PoxiaoForCausalLM(lm_config)

    # 加载预训练权重
    weight_path = args.from_weight
    if not os.path.isabs(weight_path):
        weight_path = os.path.join(args.save_dir, "..", weight_path)
    state_dict = torch.load(weight_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict, strict=False)
    Logger(f"加载预训练权重: {weight_path}")

    # LoRA
    if args.use_lora:
        model = apply_lora_to_model(model, args.lora_r, args.lora_alpha, args.lora_dropout)
        if args.lora_weights:
            load_lora_weights(model, args.lora_weights)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    Logger(f"参数: {total_params:.2f}M 总, {trainable:.2f}M 可训练")

    model = model.to(args.device)

    # 数据
    train_ds = SFTDataset(args.data_path, tokenizer._tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    val_loader = None
    if args.val_path and os.path.exists(args.val_path) and is_main():
        val_ds = SFTDataset(args.val_path, tokenizer._tokenizer, max_length=args.max_seq_len)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
        Logger(f"验证集: {len(val_ds)} 样本")

    # 优化器 (仅优化可训练参数)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))

    # DDP
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    Logger(f"训练集: {len(train_ds)} | Steps: {len(train_loader)}")

    # 训练循环
    best_loss = float("inf")
    for epoch in range(args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        model.train()

        for step, (input_ids, labels) in enumerate(train_loader, start=1):
            input_ids = input_ids.to(args.device)
            labels = labels.to(args.device)

            lr = get_lr(step, len(train_loader), args.learning_rate, warmup_steps=args.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            with torch.cuda.amp.autocast(enabled=(args.dtype == "bfloat16"), dtype=torch.bfloat16):
                res = model(input_ids, labels=labels)
                loss = res.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if step % args.log_interval == 0:
                Logger(f"[Epoch {epoch+1} Step {step}/{len(train_loader)}] loss:{loss.item():.4f} lr:{lr:.2e}")

            if args.eval_interval and step % args.eval_interval == 0 and val_loader and is_main():
                val_loss, val_ppl = evaluate(model, val_loader, args.device)
                Logger(f"  [Eval] val_loss:{val_loss:.4f} val_ppl:{val_ppl:.2f}")

                if val_loss < best_loss:
                    best_loss = val_loss
                    raw = model.module if isinstance(model, DistributedDataParallel) else model
                    torch.save(raw.state_dict(), f"{args.save_dir}/sft_best.pth")
                    if args.use_lora:
                        save_lora_weights(raw, f"{args.save_dir}/lora_best.pth")
                    Logger(f"  ✓ 保存最优模型 (val_loss={val_loss:.4f})")

            del input_ids, labels, res, loss

    # 最终保存
    if is_main():
        raw = model.module if isinstance(model, DistributedDataParallel) else model
        torch.save(raw.state_dict(), f"{args.save_dir}/sft_final.pth")
        if args.use_lora:
            save_lora_weights(raw, f"{args.save_dir}/lora_final.pth")

    if dist.is_initialized():
        dist.destroy_process_group()
    Logger("SFT 训练完成!")
