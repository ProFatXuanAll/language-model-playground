r"""Helper function for calculating perplexity.

Usage:
    import lmp.util

    perplexities = lmp.util.batch_perplexity_eval(...)
    perplexity = lmp.util.perplexity_eval(...)
"""



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Iterable
from typing import List
from typing import Union



import torch

from tqdm import tqdm



import lmp.model
import lmp.tknzr


@torch.no_grad()
def perplexity_eval(
        device: torch.device,
        model: Union[lmp.model.BaseRNNModel, lmp.model.BaseResRNNModel],
        sequence: str,
        tokenizer: lmp.tknzr.BaseTknzr
) -> float:
    r"""Helper function for calculating perplexity.

    \exp\biggl({\frac{-1}{n}\sum_{i=1}^n\log p(w_i)}\biggr)

    Args:
        device:
            Model running device.
        model:
            Language model.
        sequence:
            Sequence for evaluation. Must not be empty.
        tokenizer:
            Tokenizer for encoding sequence.

    Raises:
        TypeError:
            When one of the arguments are not an instance of their type
            annotation respectively.

    Return:
        Perplexity of `sequence`.
    """
    # Type check.
    if not isinstance(device, torch.device):
        raise TypeError('`device` must be an instance of `torch.device`.')

    if not isinstance(model, (
            lmp.model.BaseRNNModel,
            lmp.model.BaseResRNNModel,
            lmp.model.TransformerModel
    )):
        raise TypeError(
            '`model` must be an instance of '
            '`Union[lmp.model.BaseRNNModel, lmp.model.BaseResRNNModel]`.'
        )

    if not isinstance(sequence, str):
        raise TypeError('`sequence` must be an instance of `str`.')

    if not isinstance(tokenizer, lmp.tknzr.BaseTknzr):
        raise TypeError(
            '`tokenizer` must be an instance of `lmp.tknzr.BaseTknzr`.'
        )

    # Value check.
    if not sequence:
        raise ValueError('`sequence` must not be empty.')

    # Evalation mode.
    model.eval()
    model = model.to(device)

    # Encode sequence and convert into tensor. Original sequence length: S.
    # New sequence length: S + 2.
    sequence = tokenizer.encode(sequence, max_seq_len=-1)

    # `sequence[:-2]` means predict tokens include [bos] output but exclude
    # [eos] input. `x.shape = (S)`.
    x = torch.LongTensor(sequence[:-2]).to(device)

    # `y.shape = (S)`.
    y = sequence[1:-1]

    # Reshape into `(1, S)` to fit model.
    x = x.reshape(1, -1)

    # Get model vocabulary prediction with shape `(1, S, V)`.
    pred_y = model.predict(x)

    # Reshape into `(S)` for easier maniplation.
    x = x.squeeze(0)

    # Reshape into `(S, V)` for easier maniplation.
    pred_y = pred_y.squeeze(0)

    # Accumulate negative log-likelihood.
    nll = torch.zeros(1).to(device)

    # Iterate through each prediction.
    for pos, token_id in enumerate(y):
        probs = pred_y[pos, token_id]
        nll = nll - torch.log(probs)

    # Normalized by length.
    nll = nll / x.size(0)

    # Take exponential to cancel logarithmic.
    return nll.exp().item()


def batch_perplexity_eval(
        dataset: Iterable[str],
        device: torch.device,
        model: Union[lmp.model.BaseRNNModel, lmp.model.BaseResRNNModel],
        tokenizer: lmp.tknzr.BaseTknzr
) -> List[float]:
    r"""Helper function for calculating dataset perplexity.

    Args:
        dataset:
            Evaluating each sequence in the dataset. No sequences in dataset
            should be empty.
        device:
            Model running device.
        model:
            Language model.
        tokenizer:
            Tokenizer for encoding sequence.

    Raises:
        TypeError:
            When one of the arguments are not an instance of their type
            annotation respectively.

    Return:
        Perplexity of `dataset`.
    """
    # Type check.
    if not isinstance(dataset, Iterable):
        raise TypeError('`dataset` must be an instance of `Iterable[str]`.')

    perplexties = []

    try:
        for sequence in tqdm(dataset, desc='Calculating perplexities'):
            perplexity = perplexity_eval(
                device=device,
                model=model,
                sequence=sequence,
                tokenizer=tokenizer
            )
            perplexties.append(perplexity)

        return perplexties
    except TypeError as err:
        if 'sequence' in err.args[0]:
            raise TypeError(
                '`dataset` must be an instance of `Iterable[str]`.'
            )

        raise err
    except ValueError as err:
        raise ValueError(
            '`dataset` must not contain empty sequences.'
        )
