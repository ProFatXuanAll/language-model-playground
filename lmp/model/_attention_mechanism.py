r"""Attention function.

Usage:
    import lmp

    ht = lmp.model.attention_mechanism(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# 3rd-party modules

import math
import torch
import torch.nn


def attention_mechanism(
    query: torch.tensor,
    key: torch.tensor,
    value: torch.tensor
):

    device = query.device

    seq_len = query.size(1)
    d_k = query.size(-1)

    # scores 維度: (B, S, S)
    scores = query.matmul(key.transpose(-1, -2)) / math.sqrt(d_k)

    # 建立 mask 上三角矩陣遮罩
    # mask 維度: (B, S, S)
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool).to(device)
    mask = torch.triu(mask, diagonal=1)
    mask = mask.unsqueeze(0)  # (1, S, S)
    # 將未來部份的資訊遮住
    scores.masked_fill_(mask, -1e9)

    # a_matrix 維度: (B, S, S)
    a_matrix = torch.nn.functional.softmax(scores, dim=2)

    # return 維度: (B, S, H)
    return a_matrix.matmul(value)
