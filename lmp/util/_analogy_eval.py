r"""Helper function for calculating word analogy accuracy.

Usage:
    import lmp.util

    word_d = lmp.util.analogy_inference(...)
    acc_per_cat = lmp.util.analogy_eval(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Dict
from typing import Union

# 3rd-party modules

import torch

from sklearn.metrics import accuracy_score
from tqdm import tqdm

# self-made modules

import lmp.dataset
import lmp.model
import lmp.tokenizer


@torch.no_grad()
def analogy_inference(
        device: torch.device,
        model: Union[lmp.model.BaseRNNModel, lmp.model.BaseResRNNModel],
        tokenizer: lmp.tokenizer.BaseTokenizer,
        word_a: str,
        word_b: str,
        word_c: str
) -> str:
    r"""Generate analog word based on `word_a`, `word_b` and `word_c`.

    This function perform word analogy based on the following rule:
        `word_a` : `word_b` = `word_c` : `word_d`
    Where `word_d` is the prediction target.

    Args:
        device:
            Model running device.
        model:
            Language model.
        tokenizer:
            Converting token (including `word_a`, `word_b` and `word_c`) into
            token id and convert token id back to token (`word_d`). This is
            need since we use word embedding layer in our language model.
        word_a:
        word_b:
        word_c:
            Query words for word analogy.

    Raises:
        TypeError:
            When one of the arguments are not an instance of their type
            annotation respectively.

    Returns:
        Predict word following word analogy.
    """
    # Type check.
    if not isinstance(device, torch.device):
        raise TypeError('`device` must be an instance of `torch.device`.')

    if not isinstance(model, (
            lmp.model.BaseRNNModel,
            lmp.model.BaseResRNNModel
    )):
        raise TypeError(
            '`model` must be an instance of '
            '`Union[lmp.model.BaseRNNModel, lmp.model.BaseResRNNModel]`.'
        )

    if not isinstance(tokenizer, lmp.tokenizer.BaseTokenizer):
        raise TypeError(
            '`tokenizer` must be an instance of `lmp.tokenizer.BaseTokenizer`.'
        )

    if not isinstance(word_a, str):
        raise TypeError('`word_a` must be an instance of `str`.')

    if not isinstance(word_b, str):
        raise TypeError('`word_b` must be an instance of `str`.')

    if not isinstance(word_c, str):
        raise TypeError('`word_c` must be an instance of `str`.')

    # Evaluation mode.
    model.eval()
    model = model.to(device)

    # Convert tokens (query words) into token ids.
    word_a_id = torch.LongTensor([tokenizer.convert_token_to_id(word_a)])
    word_b_id = torch.LongTensor([tokenizer.convert_token_to_id(word_b)])
    word_c_id = torch.LongTensor([tokenizer.convert_token_to_id(word_c)])

    # Perform analogy calculation.
    # Shape: `(E)`.
    out = (
        model.emb_layer(word_b_id.to(device)) -
        model.emb_layer(word_a_id.to(device)) +
        model.emb_layer(word_c_id.to(device))
    )

    # Extend dimension since word embedding dimension is `(V, E)`,
    # Shape: `(1, E)`.
    out = out.unsqueeze(0)

    # Calculate cosine similarity.
    # Shape: `(V)`.
    pred = torch.nn.functional.cosine_similarity(
        out,
        model.emb_layer.weight,
        dim=0
    )

    # Get the token id with maximum consine similarity.
    # Shape: `(1)`.
    word_d_id = pred.argmax(dim=0).to('cpu')[0].item()

    # Convert back to token.
    return tokenizer.convert_id_to_token(word_d_id)


def analogy_eval(
        dataset: lmp.dataset.AnalogyDataset,
        device: torch.device,
        model: Union[lmp.model.BaseRNNModel, lmp.model.BaseResRNNModel],
        tokenizer: lmp.tokenizer.BaseTokenizer
) -> Dict[str, float]:
    r"""Helper function for calculating word analogy dataset accuracy.

    Args:
        device:
            Model running device.
        model:
            Language model.
        tokenizer:
            Converting token (including `word_a`, `word_b` and `word_c` of each
            sample in `dataset`) into token id and convert token id back to
            token (`word_d` of each sample in `dataset`). This is need since we
            use word embedding layer in our language model.
        dataset:
            Word analogy dataset.

    Raises:
        TypeError:
            When one of the arguments are not an instance of their type
            annotation respectively.

    Returns:
        A `dict` which keys are the names of each category in word analogy
        dataset, and values are the word analogy accuracy of each category.
        Additionally this function also return total accuracy of all categories
        under the key `'total'`.
    """
    # Type check.
    if not isinstance(dataset, lmp.dataset.AnalogyDataset):
        raise TypeError(
            '`dataset` must be an instance of `lmp.dataset.AnalogyDataset`'
        )

    # Save each `word_d` and `pred_word_d` for their respective category
    # accuracy and total accuracy.
    pred_per_cat = {
        'total': {
            'ans': [],
            'pred': [],
        }
    }
    for word_a, word_b, word_c, word_d, category in tqdm(dataset):
        if category not in pred_per_cat:
            pred_per_cat[category] = {
                'ans': [],
                'pred': [],
            }

        pred_per_cat[category]['ans'].append(word_d)
        pred_per_cat['total']['ans'].append(word_d)

        pred_word_d = analogy_inference(
            device=device,
            model=model,
            tokenizer=tokenizer,
            word_a=word_a,
            word_b=word_b,
            word_c=word_c
        )

        pred_per_cat[category]['pred'].append(pred_word_d)
        pred_per_cat['total']['pred'].append(pred_word_d)

    acc_per_cat = {}

    for category in pred_per_cat:
        acc_per_cat[category] = accuracy_score(
            pred_per_cat[category]['ans'],
            pred_per_cat[category]['pred'],
        )

    return acc_per_cat
