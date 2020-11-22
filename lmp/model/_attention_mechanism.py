r"""Helper function for attention mechanism.

Usage:
    import lmp

    ht = lmp.model.attention_mechanism(...)
"""


import math

import torch
import torch.nn


def attention_mechanism(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor
):
    r"""Helper function for attention mechanism.

    Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V

    First, we compute the similarity between the queries and keys. Then we
    masked future information by retaining lower triangle matrix. Finally we
    use softmax to perform normalization and multiply with values to get
    weighted sum.

    Args:
        query:
            Current vector to compute the similarity with each key vectors.
        key:
            The set of key vectors.
        value:
            The set of value vectors to compute the weighted sum.

    Return:
        A weighted sum of the values.
    """

    device = query.device

    seq_len = query.size(-2)
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

    # attn_score 維度: (B, S, S)
    attn_score = torch.nn.functional.softmax(scores, dim=-1)

    # return 維度: (B, S, H)
    return attn_score.matmul(value)
