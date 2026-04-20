"""
Poxiao Tokenizer — 基于 HuggingFace AutoTokenizer
兼容 MiniMind 的 6400 词表 byte-level BPE
"""
from transformers import AutoTokenizer

TOKENIZER_PATH = "/mnt/data2/zcz/foundation_model_research/research_repo/minimind-master/model"


class PoxiaoTokenizer:
    """Poxiao 分词器 (HF AutoTokenizer 封装)

    特性:
    - 兼容 MiniMind 6400 词表 byte-level BPE
    - 无 UNK token (byte-level 编码所有字符)
    - 支持完整的 chat template
    """

    def __init__(self, tokenizer_path=None):
        path = tokenizer_path or TOKENIZER_PATH
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        self.vocab_size = self._tokenizer.vocab_size
        self.pad_token_id = self._tokenizer.pad_token_id or 0
        self.bos_token_id = self._tokenizer.bos_token_id
        self.eos_token_id = self._tokenizer.eos_token_id

    def encode(self, text, add_special_tokens=False, max_length=None, truncation=False):
        enc = self._tokenizer(text, add_special_tokens=add_special_tokens, max_length=max_length, truncation=truncation)
        return enc.input_ids

    def decode(self, ids, skip_special_tokens=True):
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def __call__(self, *args, **kwargs):
        return self._tokenizer(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        return self._tokenizer.apply_chat_template(*args, **kwargs)
