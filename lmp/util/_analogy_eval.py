# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re

from typing import Dict
from typing import List
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
    r"""Input `word_a`, `word_b`, `word_c` to generate analogy text.
    `word_a` : `word_b` = `word_c` : `pred_word`

    Args:
        device:
            Model running device.
        model:
            Language model.
        tokenizer:
            For convert word to id or id to word.
        word_a:
            Input_data.
        word_b:
            Input_data.
        word_c:
            Input_data.

    Raises:
        TypeError:
            When one of the arguments are not an instance of their type
            annotation respectively.

    Returns:
        Predict word.
    """
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
            '`tokenizer` must be an instance of '
            '`lmp.tokenizer.BaseTokenizer`.'
        )
    if not isinstance(word_a, str):
        raise TypeError('`word_a` must be an instance of `str`.')
    if not isinstance(word_b, str):
        raise TypeError('`word_b` must be an instance of `str`.')
    if not isinstance(word_c, str):
        raise TypeError('`word_c` must be an instance of `str`.')

    #(E, V)
    emb = model.emb_layer.weight.transpose(0, 1)

    # Syntatic and semaintic test.
    word_a_id = torch.tensor(tokenizer.convert_token_to_id(word_a))
    word_b_id = torch.tensor(tokenizer.convert_token_to_id(word_b))
    word_c_id = torch.tensor(tokenizer.convert_token_to_id(word_c))

    # (E)
    out = (
        model.emb_layer(word_b_id.to(device)) -
        model.emb_layer(word_a_id.to(device)) +
        model.emb_layer(word_c_id.to(device))
    )

    # (E, 1)
    out = out.unsqueeze(-1)

    # (V)
    pred = torch.nn.functional.cosine_similarity(out, emb, dim=0)

    # (B)
    pred = pred.argmax(dim=-1).to('cpu').tolist()

    # Convert back to sequence.
    pred_word = tokenizer.convert_id_to_token(pred)
    return pred_word


@torch.no_grad()
def analogy_eval(
    dataset: lmp.dataset.AnalogyDataset,
    device: torch.device,
    model: Union[lmp.model.BaseRNNModel, lmp.model.BaseResRNNModel],
    tokenizer: lmp.tokenizer.BaseTokenizer
) -> Dict[str, float]:
    r"""Use specified data set to calculate analogy test score.

    Args:
        device:
            Model running device.
        model:
            Language model.
        tokenizer:
            For convert word to id or id to word.
        dataset:
            Test data.

    Raises:
        TypeError:
            When one of the arguments are not an instance of their type
            annotation respectively.

    Returns:
        acc_per_cat:
            A dictionary whose key is the name of each category.
    """
    if not isinstance(dataset, lmp.dataset.AnalogyDataset):
        raise TypeError(
            '`dataset` must be an instance of `lmp.dataset.AnalogyDataset`'
        )
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
            '`tokenizer` must be an instance of '
            '`lmp.tokenizer.BaseTokenizer`.'
        )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32
    )

    pred_per_cat = {}
    for word_a, word_b, word_c, word_d, categorys in tqdm(dataloader):
        for category in categorys:
            if category not in pred_per_cat:
                pred_per_cat[category] = {
                    'ans': [],
                    'pred': [],
                }

        for i in range(len(word_d)):
            pred_per_cat[categorys[i]]['ans'].append(word_d[i])
            pred_per_cat[categorys[i]]['pred'].append(
                analogy_inference(
                    device=device,
                    model=model,
                    tokenizer=tokenizer,
                    word_a=word_a[i],
                    word_b=word_b[i],
                    word_c=word_c[i]
                )
            )

    acc_per_cat = {}

    for category in pred_per_cat:
        acc_per_cat[category] = accuracy_score(
            pred_per_cat[category]['ans'],
            pred_per_cat[category]['pred'],
        )

    return acc_per_cat
