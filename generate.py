"""
Poxiao (破晓) 文本生成脚本
"""
import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PoxiaoConfig
from model import PoxiaoForCausalLM
from tokenizer import PoxiaoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Poxiao Text Generation")
    parser.add_argument("--weight", type=str, required=True, help="模型权重路径")
    parser.add_argument("--prompt", type=str, default="从前有座山，", help="输入提示文本")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.85, help="采样温度")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p 采样阈值")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # 加载模型
    config = PoxiaoConfig()
    model = PoxiaoForCausalLM(config).to(args.device)
    state_dict = torch.load(args.weight, map_location=args.device)
    model.load_state_dict(state_dict)
    model.eval()

    # 加载 tokenizer
    tokenizer = PoxiaoTokenizer()

    # 编码 prompt
    ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    input_ids = torch.tensor([ids], dtype=torch.long, device=args.device)
    print(f"Prompt: {args.prompt}")
    print(f"Tokens: {len(ids)}")

    # 生成
    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=tokenizer.eos_token_id,
    )

    # 解码
    text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
    print(f"\n{'='*60}")
    print(f"Generated ({output_ids.shape[1] - len(ids)} new tokens):")
    print(f"{'='*60}")
    print(text)


if __name__ == "__main__":
    main()
