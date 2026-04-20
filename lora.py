"""
Poxiao LoRA — 纯 PyTorch 实现的 LoRA 微调
仅作用于 Attention QKV 投影层, 冻结主干权重
"""
import math
import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    """通用 LoRA 适配器: W' = W + BA * alpha/r"""

    def __init__(self, original_linear: nn.Linear, r: int = 8, lora_alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.original = original_linear
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # 冻结原始权重
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

        # LoRA 矩阵
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        self.dropout = nn.Dropout(dropout)

        # 初始化: A 用 kaiming, B 用零
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        original_out = self.original(x)
        lora_out = (self.dropout(x) @ self.lora_A @ self.lora_B) * self.scaling
        return original_out + lora_out


def apply_lora_to_model(model, r: int = 8, lora_alpha: int = 16, dropout: float = 0.05):
    """对模型的 Attention QKV 投影层应用 LoRA"""
    lora_count = 0

    for layer in model.model.layers:
        attn = layer.self_attn

        # QKV 投影层
        for name in ["q_proj", "k_proj", "v_proj"]:
            original = getattr(attn, name)
            if isinstance(original, nn.Linear):
                lora_layer = LoRALayer(original, r, lora_alpha, dropout)
                setattr(attn, name, lora_layer)
                lora_count += 1

    # 冻结所有非 LoRA 参数
    for name, param in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA: {lora_count} 层适配, 可训练 {trainable/1e6:.2f}M / {total/1e6:.2f}M ({trainable/total*100:.2f}%)")

    return model


def get_lora_state_dict(model):
    """提取 LoRA 权重"""
    lora_dict = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_dict[name] = param.data.cpu().half()
    return lora_dict


def save_lora_weights(model, path):
    """保存 LoRA 权重"""
    lora_dict = get_lora_state_dict(model)
    torch.save(lora_dict, path)
    print(f"LoRA 权重保存: {path} ({len(lora_dict)} 参数)")


def load_lora_weights(model, path):
    """加载 LoRA 权重"""
    lora_dict = torch.load(path, map_location="cpu")
    for name, param in model.named_parameters():
        if name in lora_dict:
            param.data.copy_(lora_dict[name].to(param.device))
    print(f"LoRA 权重加载: {path} ({len(lora_dict)} 参数)")
