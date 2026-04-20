"""
Poxiao 数据集 — 基于 HuggingFace datasets 库
支持自动缓存、流式加载、分布式采样
"""
import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class PretrainDataset(Dataset):
    """预训练数据集 (datasets.load_dataset)

    - 自动缓存: datasets 库自动缓存处理后的数据
    - 分布式兼容: 配合 DistributedSampler 实现多卡数据分片
    - 内存高效: datasets 使用 memory-mapped Arrow 格式
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 768):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id or 0
        # 使用 datasets.load_dataset 自动缓存 + 高效内存映射
        self.samples = load_dataset("json", data_files=data_path, split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        tokens = self.tokenizer(
            str(sample["text"]),
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True,
        ).input_ids
        tokens = [self.bos_id] + tokens + [self.eos_id]
        pad_len = self.max_length - len(tokens)
        input_ids = tokens + [self.pad_id] * pad_len
        labels = input_ids.copy()
        for i in range(len(tokens), self.max_length):
            labels[i] = -100
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class SFTDataset(Dataset):
    """SFT 数据集 — 支持两种格式:
    1. 对话格式: {"conversations": [{"role": "user", "content": "..."}, ...]}
    2. 纯文本格式: {"text": "<s>Human: ...\\nAssistant: ...</s>"}
    """

    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=jsonl_path, split="train")
        # 检测格式
        sample = self.samples[0]
        self.format = "text" if "text" in sample.features else "conversations"
        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = [dict(msg) for msg in conversations]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    def generate_labels(self, input_ids):
        """对话格式: 仅在 assistant 回复部分计算 loss"""
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def generate_text_labels(self, input_ids):
        """纯文本格式: Assistant 之后的内容计算 loss"""
        labels = [-100] * len(input_ids)
        # 找 "Assistant:" 的位置
        assistant_ids = self.tokenizer("Assistant:", add_special_tokens=False).input_ids
        for i in range(len(input_ids) - len(assistant_ids)):
            if input_ids[i:i + len(assistant_ids)] == assistant_ids:
                # Assistant 之后的内容计算 loss
                for j in range(i + len(assistant_ids), len(input_ids)):
                    labels[j] = input_ids[j]
                break
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        if self.format == "text":
            text = sample["text"]
            input_ids = self.tokenizer(text, add_special_tokens=False).input_ids[: self.max_length]
            pad_len = self.max_length - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id or 0] * pad_len
            labels = self.generate_text_labels(input_ids)
        else:
            prompt = self.create_chat_prompt(sample["conversations"])
            input_ids = self.tokenizer(prompt).input_ids[: self.max_length]
            pad_len = self.max_length - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id or 0] * pad_len
            labels = self.generate_labels(input_ids)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
