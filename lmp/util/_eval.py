r"""Helper function for calculating perplexity.

Usage:
    import lmp

    generated = lmp.util.batch_eval(...)
    generated = lmp.util.eval(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import List

# 3rd-party modules

import torch

# self-made modules

import lmp

@torch.no_grad()
def eval(
    device: torch.device,
    model: lmp.model.BaseRNNModel,
    sequence: str,
    tokenizer: lmp.tokenizer.BaseTokenizer
) -> torch.Tensor:
    r"""
    Helper function for calculating perplexity.

    Args:
        device:
            Model running device.
        model:
            Language model.
        sequence:
            Test seqence for evaluating.
        tokenizer:
            Tokenizer for encoding sequence.

    Return:
        Perplexity of test sequence.
        dtype = float32
    """

    model.eval()

    sequence = tokenizer.encode(sequence, max_seq_len=-1)
    sequence = torch.LongTensor(sequence).to(device)

    seq_len = sequence.size(0)

    sequence = sequence.view(1, seq_len)

    pred_y = model.predict(sequence)    # (1, S, V)

    sequence = sequence.squeeze(0)
    pred_y = pred_y.squeeze(0)

    seq_prob = torch.zeros(1)
    for i in range(seq_len - 1):
        word_probs = pred_y[i]
        word_id = sequence[i+1]
        seq_prob = seq_prob - torch.log(word_probs[word_id])

    seq_prob = seq_prob / (seq_len - 1)

    return seq_prob.exp()


def batch_eval(
    dataset: List[str],
    device: torch.device,
    model: lmp.model.BaseRNNModel,
    tokenizer: lmp.tokenizer.BaseTokenizer
):
    r"""
    Helper function for calculating test dataset perplexity.

    Args:
        dataset:
            Test dataset for evaluating each sequence.
        device:
            Model running device.
        model:
            Language model.
        tokenizer:
            Tokenizer for encoding sequence.

    """
    for sequence in dataset:
        perplexity = eval(
            device=device,
            model=model,
            sequence=sequence,
            tokenizer=tokenizer
        )
        print(f'{perplexity}, {sequence}')
