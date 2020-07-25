r"""Helper function for sequences generation.

Usage:
    import lmp

    generated = lmp.util.generate_sequence(...)
    generated = lmp.util.generate_sequence_by_config(...)
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
def generate_sequence(
        beam_width: int,
        begin_of_sequence: str,
        device: torch.device,
        max_seq_len: int,
        model: lmp.model.BaseRNNModel,
        tokenizer: lmp.tokenizer.BaseTokenizer
) -> List[str]:
    r"""Helper function for sequences generation.

    Args:
        beam_width:
            Number of candidate sequences to output.
        begin_of_sequence:
            Begining of sequence which model will auto-complete.
        device:
            Model running device.
        max_seq_len:
            Maximum of output sequences length.
        model:
            Language model.
        tokenizer:
            Tokenizer for encoding and decoding sequences.

    Returns:
        Generated sequences.
    """

    model.eval()

    sequence = tokenizer.encode(begin_of_sequence, max_seq_len=-1)
    input_seq = torch.LongTensor(sequence)[:-1].to(device)

    generated_sequences = model.generate(
        begin_of_sequence=input_seq,
        beam_width=beam_width,
        max_seq_len=max_seq_len
    )

    return tokenizer.batch_decode(generated_sequences.tolist())


def generate_sequence_by_config(
        beam_width: int,
        begin_of_sequence: str,
        config: lmp.config.BaseConfig,
        max_seq_len: int,
        model: lmp.model.BaseRNNModel,
        tokenizer: lmp.tokenizer.BaseTokenizer
) -> List[str]:
    r"""Helper function for sequences generation.

    Args:
        beam_width:
            Number of candidate sequences to output.
        begin_of_sequence:
            Begining of sequence which model will auto-complete.
        config:
            Configuration object with attributes `device`.
        max_seq_len:
            Maximum of output sequences length.
        model:
            Language model.
        tokenizer:
            Tokenizer for encoding and decoding sequences.

    Returns:
        Generated sequences.
    """

    return generate_sequence(
        beam_width=beam_width,
        begin_of_sequence=begin_of_sequence,
        device=config.device,
        max_seq_len=max_seq_len,
        model=model,
        tokenizer=tokenizer
    )
