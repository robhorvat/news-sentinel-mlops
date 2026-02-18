from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        num_filters: int = 128,
        kernel_sizes: Sequence[int] = (3, 4, 5),
        num_classes: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embedding_dim, num_filters, kernel_size=k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)  # [B, L, E]
        x = embedded.transpose(1, 2)  # [B, E, L]

        pooled = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))  # [B, F, L-k+1]
            pooled_out = F.max_pool1d(conv_out, kernel_size=conv_out.shape[2]).squeeze(2)
            pooled.append(pooled_out)

        features = torch.cat(pooled, dim=1)
        features = self.dropout(features)
        return self.classifier(features)
