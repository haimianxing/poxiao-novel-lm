"""
Poxiao (破晓) 预训练脚本 v3
完整支持: DDP 分布式、验证集评估、早停、分阶段窗口(768→2048)、多任务混合
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
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PoxiaoConfig
from model import PoxiaoForCausalLM
from tokenizer import PoxiaoTokenizer
from dataset import PretrainDataset

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════
def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(msg):
    if is_main():
        print(msg, flush=True)


def get_lr(step, total_steps, max_lr, min_lr=1e-5, warmup_steps=100):
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


# ═══════════════════════════════════════════════════════════════
#  分阶段训练常量
#  Phase A (0-950):    seq_len=768,  纯 CLM
#  Phase B (950-1400): seq_len=2048, 多任务 0→8%
#  Phase C (1400+):    seq_len=2048, 多任务 10%
# ═══════════════════════════════════════════════════════════════
PHASE_A_END = 950
PHASE_B_END = 1400


def compute_weighted_loss(logits, labels, long_range_weight=2.0, threshold=1024):
    """长程损失加权: 前 threshold 个 token 权重 1.0, 之后权重 long_range_weight"""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    B, T, V = shift_logits.shape
    # 逐 token CE loss
    per_token_loss = F.cross_entropy(
        shift_logits.view(-1, V), shift_labels.view(-1), ignore_index=-100, reduction='none'
    ).view(B, T)
    # 位置权重
    weights = torch.ones(T, device=labels.device)
    if T > threshold:
        weights[threshold:] = long_range_weight
    weights = weights.unsqueeze(0)  # [1, T]
    # 有效 token mask
    mask = (shift_labels != -100).float()
    weighted = (per_token_loss * weights * mask).sum() / mask.sum().clamp(min=1)
    return weighted


def init_distributed():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0
    dist.init_process_group(backend="gloo")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


# ═══════════════════════════════════════════════════════════════
#  断点续训: 保存/加载
# ═══════════════════════════════════════════════════════════════
def save_checkpoint(model, optimizer, scaler, epoch, step, global_step, config, save_dir, wandb=None):
    os.makedirs(save_dir, exist_ok=True)
    raw = model.module if isinstance(model, DistributedDataParallel) else model
    raw = getattr(raw, "_orig_mod", raw)

    moe_s = "_moe" if config.use_moe else ""
    weight_path = f"{save_dir}/pretrain_{config.hidden_size}{moe_s}_step{global_step}.pth"
    tmp = weight_path + ".tmp"
    state = {k: v.half().cpu() for k, v in raw.state_dict().items()}
    torch.save(state, tmp)
    os.replace(tmp, weight_path)

    # 完整训练状态
    resume_path = f"{save_dir}/pretrain_{config.hidden_size}{moe_s}_resume.pth"
    tmp = resume_path + ".tmp"
    resume_data = {
        "model": state,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "step": step,
        "global_step": global_step,
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "config": config.to_dict(),
    }
    torch.save(resume_data, tmp)
    os.replace(tmp, resume_path)
    del state, resume_data
    torch.cuda.empty_cache()
    return weight_path


def load_checkpoint(config, save_dir):
    moe_s = "_moe" if config.use_moe else ""
    resume_path = f"{save_dir}/pretrain_{config.hidden_size}{moe_s}_resume.pth"
    if os.path.exists(resume_path):
        data = torch.load(resume_path, map_location="cpu")
        saved_ws = data.get("world_size", 1)
        current_ws = dist.get_world_size() if dist.is_initialized() else 1
        if saved_ws != current_ws:
            data["step"] = data["step"] * saved_ws // current_ws
        return data
    return None


# ═══════════════════════════════════════════════════════════════
#  验证集评估
# ═══════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=None):
    """计算验证集 loss 和 perplexity"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i, (input_ids, labels) in enumerate(val_loader):
        if max_batches and i >= max_batches:
            break
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        res = model(input_ids, labels=labels)
        n_tokens = (labels != -100).sum().item()
        total_loss += res.loss.item() * n_tokens
        total_tokens += n_tokens
        del input_ids, labels, res

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20))  # clamp to avoid overflow
    model.train()
    return avg_loss, perplexity


# ═══════════════════════════════════════════════════════════════
#  训练循环
# ═══════════════════════════════════════════════════════════════
def train_loop(model, train_loader, mt_loader, val_loader, optimizer, scaler, config, args, wandb=None, train_sampler=None):
    best_val_ppl = float("inf")
    patience_counter = 0
    best_ckpt_path = None
    global_step = 0

    total_steps = len(train_loader) * args.epochs
    total_tokens = 0
    current_phase = "A"

    # 多任务迭代器
    mt_iter = iter(mt_loader) if mt_loader is not None else None

    model.train()
    start_time = time.time()

    for epoch in range(args.epochs):
        if dist.is_initialized() and train_sampler:
            train_sampler.set_epoch(epoch)
        Logger(f"\n--- Epoch {epoch+1}/{args.epochs} ---")

        for step, (input_ids, labels) in enumerate(train_loader, start=1):
            # ── Phase 检测 ──
            if global_step < PHASE_A_END:
                phase = "A"
            elif global_step < PHASE_B_END:
                phase = "B"
            else:
                phase = "C"

            if phase != current_phase:
                Logger(f"\n{'='*50}")
                Logger(f"  Phase {current_phase} → {phase} (step {global_step})")
                if phase == "A":
                    Logger(f"  seq_len=768, 纯 CLM")
                elif phase == "B":
                    Logger(f"  seq_len=2048, 多任务 0→8%")
                else:
                    Logger(f"  seq_len=2048, 多任务 10%")
                Logger(f"{'='*50}\n")
                current_phase = phase

            # ── Phase A: 截断到 768 ──
            if phase == "A":
                input_ids = input_ids[:, :768]
                labels = labels[:, :768]

            input_ids = input_ids.to(args.device)
            labels = labels.to(args.device)

            # ── Phase B/C: 多任务批次替换 ──
            if phase in ("B", "C") and mt_iter is not None:
                if phase == "B":
                    mt_ratio = (global_step - PHASE_A_END) / max(PHASE_B_END - PHASE_A_END, 1) * 0.08
                else:
                    mt_ratio = 0.10
                if torch.rand(1).item() < mt_ratio:
                    try:
                        mt_ids, mt_lbl = next(mt_iter)
                    except StopIteration:
                        mt_iter = iter(mt_loader)
                        mt_ids, mt_lbl = next(mt_iter)
                    input_ids = mt_ids.to(args.device)
                    labels = mt_lbl.to(args.device)

            # ── 学习率调度 ──
            lr = get_lr(global_step, total_steps, args.learning_rate, warmup_steps=args.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # ── 前向 + 长程加权损失 ──
            with torch.cuda.amp.autocast(enabled=(args.dtype == "bfloat16"), dtype=torch.bfloat16):
                res = model(input_ids)  # 不传 labels, 手动计算加权损失
                ce_loss = compute_weighted_loss(res.logits, labels)
                loss = (ce_loss + res.aux_loss) / args.accumulation_steps

            # ── 反向传播 ──
            scaler.scale(loss).backward()

            if step % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            total_tokens += input_ids.numel()

            # ── 日志 ──
            if step % args.log_interval == 0 or step == len(train_loader):
                cur_loss = loss.item() * args.accumulation_steps
                cur_lr = optimizer.param_groups[-1]["lr"]
                elapsed = time.time() - start_time
                eta = elapsed / max(global_step, 1) * (total_steps - global_step) / 60
                Logger(
                    f"[Step {global_step}/{total_steps}] "
                    f"loss:{cur_loss:.4f} lr:{cur_lr:.2e} "
                    f"tokens:{total_tokens/1e6:.1f}M eta:{eta:.0f}min"
                )
                if wandb:
                    wandb.log({"loss": cur_loss, "lr": cur_lr, "step": global_step})

            # ── 验证 + 早停 ──
            if args.eval_interval and global_step % args.eval_interval == 0:
                if is_main() and val_loader is not None:
                    val_loss, val_ppl = evaluate(model, val_loader, args.device, max_batches=100)
                    Logger(f"  [Eval Step {global_step}] val_loss:{val_loss:.4f} val_ppl:{val_ppl:.2f}")

                    if wandb:
                        wandb.log({"val_loss": val_loss, "val_ppl": val_ppl, "step": global_step})

                    # 早停检查 (需先降到 min_ppl_threshold 以下)
                    if val_ppl < best_val_ppl:
                        best_val_ppl = val_ppl
                        patience_counter = 0
                        best_ckpt_path = save_checkpoint(
                            model, optimizer, scaler, 0, step, global_step, config, args.save_dir, wandb
                        )
                        import shutil
                        best_path = f"{args.save_dir}/pretrain_{config.hidden_size}_best.pth"
                        if best_ckpt_path:
                            shutil.copy2(best_ckpt_path, best_path)
                        Logger(f"  ✓ 新最优 val_ppl={val_ppl:.2f}, 已保存")
                    else:
                        patience_counter += 1
                        Logger(f"  ✗ val_ppl 未改善 ({patience_counter}/{args.early_stop_patience})")
                        if patience_counter >= args.early_stop_patience:
                            if best_val_ppl <= args.min_ppl_threshold:
                                Logger(f"早停触发! 最优 val_ppl={best_val_ppl:.2f} ≤ {args.min_ppl_threshold}")
                                return best_ckpt_path, best_val_ppl
                            else:
                                Logger(f"  val_ppl={best_val_ppl:.2f} > {args.min_ppl_threshold}, 继续训练")
                                patience_counter = 0

            # ── 定期保存 ──
            if args.save_interval and global_step % args.save_interval == 0 and is_main():
                save_checkpoint(model, optimizer, scaler, 0, step, global_step, config, args.save_dir, wandb)

            del input_ids, labels, res, loss

    # 训练结束, 最终保存
    if is_main():
        best_ckpt_path = save_checkpoint(
            model, optimizer, scaler, 0, total_steps, global_step, config, args.save_dir, wandb
        )

    return best_ckpt_path, best_val_ppl


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poxiao Pretraining v3")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1.5e-3)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=0.1)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--warmup_steps", type=int, default=800)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--min_ppl_threshold", type=float, default=15.0,
                        help="早停仅在 val_ppl 低于此阈值时生效")
    # 模型配置
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    # 数据
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--multitask_path", type=str, default=None)
    parser.add_argument("--multitask_ratio", type=float, default=0.1)
    # 恢复训练
    parser.add_argument("--from_weight", type=str, default="none")
    parser.add_argument("--from_resume", type=int, default=0, choices=[0, 1])
    # wandb
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Poxiao-Pretrain")
    # 其他
    parser.add_argument("--use_compile", type=int, default=0, choices=[0, 1])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 1. 分布式初始化
    local_rank = init_distributed()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(args.seed)

    # 2. 模型配置
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = PoxiaoConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        max_position_embeddings=args.max_seq_len,
        rope_theta=1e4,
    )
    ckp_data = load_checkpoint(lm_config, args.save_dir) if args.from_resume else None

    # 3. 混合精度
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))

    # TF32 加速
    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # 4. Wandb
    wandb = None
    if args.use_wandb and is_main():
        try:
            import swanlab as wandb_mod
        except ImportError:
            import wandb as wandb_mod
        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        wandb_mod.init(
            project=args.wandb_project,
            name=f"Poxiao-H{args.hidden_size}-BS{args.batch_size}-LR{args.learning_rate}",
            id=wandb_id,
            resume="must" if wandb_id else None,
        )
        wandb = wandb_mod

    # 5. 模型、Tokenizer
    tokenizer = PoxiaoTokenizer()
    model = PoxiaoForCausalLM(lm_config)

    if args.from_weight != "none":
        weight_path = f"{args.save_dir}/{args.from_weight}"
        model.load_state_dict(torch.load(weight_path, map_location=args.device), strict=False)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    Logger(f"Poxiao Model Params: {total_params:.2f}M")
    Logger(f"Config: rope_theta=1e4, max_seq_len={args.max_seq_len}")
    model = model.to(args.device)

    # 6. 数据
    train_ds = PretrainDataset(args.data_path, tokenizer._tokenizer, max_length=args.max_seq_len)

    # 多任务数据 (单独 DataLoader, Phase B/C 中动态混合)
    mt_loader = None
    if args.multitask_path and os.path.exists(args.multitask_path):
        Logger(f"加载多任务数据: {args.multitask_path}")
        mt_ds = PretrainDataset(args.multitask_path, tokenizer._tokenizer, max_length=args.max_seq_len)
        mt_loader = DataLoader(
            mt_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
        )
        Logger(f"多任务数据: {len(mt_ds)} 样本 (Phase B/C 动态混合)")

    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    val_loader = None
    if args.val_path and os.path.exists(args.val_path) and is_main():
        val_ds = PretrainDataset(args.val_path, tokenizer._tokenizer, max_length=args.max_seq_len)
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )
        Logger(f"验证集: {len(val_ds)} 样本")

    # 7. 优化器
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        betas=(0.9, 0.999), weight_decay=args.weight_decay,
    )

    # 8. 恢复训练状态
    start_step = 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_step = ckp_data.get("step", 0)
        Logger(f"Resumed from step={start_step}")

    # 9. torch.compile + DDP
    if args.use_compile:
        model = torch.compile(model)
        Logger("torch.compile enabled")
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    n_gpu = dist.get_world_size() if dist.is_initialized() else 1
    global_bs = args.batch_size * n_gpu
    total_steps = len(train_loader)
    Logger(f"数据集: {len(train_ds):,} 样本 | GPU: {n_gpu} | 全局BS: {global_bs} | 总步数: {total_steps}")
    Logger(f"每步 tokens: {global_bs * args.max_seq_len:,} | 总 tokens: {global_bs * args.max_seq_len * total_steps / 1e9:.2f}B")

    # 10. 训练
    best_ckpt, best_ppl = train_loop(
        model, train_loader, mt_loader, val_loader, optimizer, scaler, lm_config, args, wandb,
        train_sampler=train_sampler,
    )

    # 11. 清理
    if dist.is_initialized():
        dist.destroy_process_group()
    Logger(f"训练完成! 最优 val_ppl={best_ppl:.2f}, checkpoint={best_ckpt}")
