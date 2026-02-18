from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from news_sentinel.data.preprocess import whitespace_tokenize

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1


@dataclass
class Vocabulary:
    token_to_idx: Dict[str, int]
    idx_to_token: List[str]

    def __len__(self) -> int:
        return len(self.idx_to_token)

    def encode(self, text: str, max_length: int) -> List[int]:
        ids = [self.token_to_idx.get(tok, UNK_IDX) for tok in whitespace_tokenize(text)]
        ids = ids[:max_length]
        if len(ids) < max_length:
            ids.extend([PAD_IDX] * (max_length - len(ids)))
        return ids


def build_vocabulary(
    texts: Iterable[str], max_vocab_size: int = 30000, min_freq: int = 2
) -> Vocabulary:
    counts: Counter = Counter()
    for text in texts:
        counts.update(whitespace_tokenize(text))

    most_common = [
        token for token, freq in counts.most_common(max_vocab_size) if freq >= min_freq
    ]

    idx_to_token = [PAD_TOKEN, UNK_TOKEN] + most_common
    token_to_idx = {token: idx for idx, token in enumerate(idx_to_token)}
    return Vocabulary(token_to_idx=token_to_idx, idx_to_token=idx_to_token)


class EncodedTextDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Sequence[int], vocab: Vocabulary, max_length: int):
        self.features = [vocab.encode(t, max_length=max_length) for t in texts]
        self.labels = [int(y) for y in labels]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        feature = torch.tensor(self.features[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label


def create_dataloader(
    texts: Sequence[str],
    labels: Sequence[int],
    vocab: Vocabulary,
    max_length: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = EncodedTextDataset(texts=texts, labels=labels, vocab=vocab, max_length=max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
