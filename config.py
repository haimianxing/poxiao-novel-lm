"""
PoxiaoConfig — 破晓大模型配置
继承 HuggingFace PretrainedConfig，兼容 HF 生态
"""
from transformers import PretrainedConfig
import math


class PoxiaoConfig(PretrainedConfig):
    model_type = "poxiao"

    def __init__(
        self,
        vocab_size: int = 6400,
        hidden_size: int = 768,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        intermediate_size: int = None,
        max_position_embeddings: int = 2048,
        rope_theta: float = 1e4,
        rms_norm_eps: float = 1e-6,
        hidden_act: str = "silu",
        dropout: float = 0.0,
        tie_word_embeddings: bool = True,
        flash_attn: bool = True,
        use_moe: bool = False,
        num_experts: int = 4,
        num_experts_per_tok: int = 1,
        moe_intermediate_size: int = None,
        norm_topk_prob: bool = True,
        router_aux_loss_coef: float = 5e-4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = (
            intermediate_size
            if intermediate_size is not None
            else math.ceil(hidden_size * math.pi / 64) * 64
        )
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.dropout = dropout
        self.tie_word_embeddings = tie_word_embeddings
        self.flash_attn = flash_attn
        # MoE
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = (
            moe_intermediate_size if moe_intermediate_size is not None else self.intermediate_size
        )
        self.norm_topk_prob = norm_topk_prob
        self.router_aux_loss_coef = router_aux_loss_coef

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @property
    def num_params(self):
        h, v, l = self.hidden_size, self.vocab_size, self.num_hidden_layers
        inter = self.intermediate_size
        embed = v * h
        attn = l * (
            h * h
            + self.num_key_value_heads * self.head_dim * h * 2
            + h * h
            + self.head_dim * 2
        )
        ffn = l * h * inter * 3
        norm = h * (l * 2 + 1)
        return (embed + attn + ffn + norm) if self.tie_word_embeddings else (embed * 2 + attn + ffn + norm)
