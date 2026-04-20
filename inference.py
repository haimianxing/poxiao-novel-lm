"""
Poxiao 推理脚本 — 预训练模型文本续写测试
用法:
  # 自动测试 (预设小说 prompt)
  python inference.py --weight ../out/pretrain/pretrain_768_best.pth

  # 交互式输入
  python inference.py --weight ../out/pretrain/pretrain_768_best.pth --mode 1

  # 调整生成长度和温度
  python inference.py --weight ../out/pretrain/pretrain_768_best.pth --max_new_tokens 512 --temperature 0.7
"""
import os
import sys
import time
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PoxiaoConfig
from model import PoxiaoForCausalLM
from tokenizer import PoxiaoTokenizer


def load_model(weight_path, device="cuda"):
    config = PoxiaoConfig(
        hidden_size=768,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=2048,
        rope_theta=1e4,
    )
    model = PoxiaoForCausalLM(config)

    state_dict = torch.load(weight_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict, strict=False)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Poxiao {total_params:.2f}M params loaded from {weight_path}")
    return model.half().eval().to(device)


PROMPTS_PRETRAIN = [
    # 小说续写
    "清晨的阳光透过窗帘洒在书桌上，李明拿起那封已经泛黄的信，",
    "武林中流传着一个古老的传说，",
    "她站在城楼上，望着远方的烽火，心中涌起一股难以名状的悲痛。",
    "这座城市已经沉睡了千年，直到今天，",
    # 多任务标签续写
    "[文风匹配] 目标文风：玄幻\n正文：少年凌风站在悬崖之上，",
    "[章节结构] 本章叙事阶段：开端\n正文：\n第一回",
    "[叙事节奏] 本段节奏：舒缓铺垫\n正文：春日的午后，",
    "[全本大纲构建] 核心设定：",
]


@torch.inference_mode()
def generate(model, tokenizer, prompt, max_new_tokens=256, temperature=0.85,
             top_p=0.85, top_k=50, device="cuda"):
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    start = time.time()
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
    )
    elapsed = time.time() - start

    new_tokens = output_ids.shape[1] - input_ids.shape[1]
    generated = tokenizer.decode(output_ids[0].cpu().tolist()[input_ids.shape[1]:],
                                  skip_special_tokens=True)
    speed = new_tokens / max(elapsed, 0.001)
    return generated, new_tokens, speed


def main():
    parser = argparse.ArgumentParser(description="Poxiao Inference")
    parser.add_argument("--weight", type=str, required=True, help="模型权重路径")
    parser.add_argument("--mode", type=int, default=0, help="0=自动测试, 1=交互输入")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model = load_model(args.weight, args.device)
    tokenizer = PoxiaoTokenizer()

    print(f"\n{'='*60}")
    print(f"Poxiao 预训练模型推理测试")
    print(f"  温度: {args.temperature} | top_p: {args.top_p} | top_k: {args.top_k}")
    print(f"  最大生成长度: {args.max_new_tokens} tokens")
    print(f"{'='*60}\n")

    if args.mode == 0:
        # 自动测试
        for i, prompt in enumerate(PROMPTS_PRETRAIN):
            text, n_tok, speed = generate(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                device=args.device,
            )
            print(f"[Prompt {i+1}] {prompt}")
            print(f"[生成 {n_tok} tokens, {speed:.1f} tok/s]")
            print(f"{text}")
            print("-" * 60)
    else:
        # 交互式
        print("输入 prompt 进行续写 (空行退出):\n")
        while True:
            try:
                prompt = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not prompt:
                break
            text, n_tok, speed = generate(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                device=args.device,
            )
            print(f"\n[生成 {n_tok} tokens, {speed:.1f} tok/s]")
            print(f"{prompt}{text}")
            print("-" * 60)


if __name__ == "__main__":
    main()
