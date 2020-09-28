r"""Helper function for attention mechanism.

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
    r"""Helper function for attention mechanism.

    Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{n}})V

    First, we compute the similarity between the query and key to obtain scores.
    Second, mask future information of scores by retaining lower triangle
    matrix. Third, use a softmax function to normalize the scores. And finally
    scores multiply the corresponding values to get output vector.

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
