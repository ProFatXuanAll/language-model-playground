# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re

from typing import List

# 3rd-party modules

import torch

from sklearn.metrics import accuracy_score
from tqdm import tqdm

# self-made modules

import lmp


@torch.no_grad()
def calc_correct_percent(
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        model: lmp.model.BaseRNNModel,
        tokenizer: lmp.tokenizer.BaseTokenizer
) -> float:
    """ To calculate correct percentage of this model
    in the test data.

    Args:
        embed_table:
            The table map word id into embedding form.
        data_loader:
            To get test data.
        device:
            Compute running device.
        tokenizer:
            To convert word id to word or word to word id.
    """

    # (1, E, V)
    emb = model.embedding_layer.weight.transpose(0, 1).unsqueeze(0)

    # Record prediction result.
    pred_per_cat = {}

    # Syntatic and semaintic test.
    # (B)
    for word_a_id, word_b_id, word_c_id, word_d, category in tqdm(data_loader):
        # (B, E)
        out = (
            model.embedding_layer(word_b_id.to(device)) -
            model.embedding_layer(word_a_id.to(device)) +
            model.embedding_layer(word_c_id.to(device))
        )

        # (B, E, 1)
        out = out.unsqueeze(-1)

        # (B, V)
        pred = torch.nn.functional.cosine_similarity(out, emb, dim=1)

        # (B)
        pred = pred.argmax(dim=-1).to('cpu').tolist()

        # Convert back to sequence.
        pred = tokenizer.convert_ids_to_tokens(pred)

        for i, ans_word in enumerate(word_d):
            if category[i] not in pred_per_cat:
                pred_per_cat[category[i]] = {
                    'pred': [],
                    'ans': [],
                }
            pred_per_cat[category[i]]['pred'].append(pred[i])
            pred_per_cat[category[i]]['ans'].append(ans_word)

    acc_per_cat = {}
    for category in pred_per_cat:
        acc = accuracy_score(
            pred_per_cat[category]['pred'],
            pred_per_cat[category]['ans']
        )
        acc_per_cat[category] = acc

    all_pred = []
    all_ans = []
    for category in pred_per_cat:
        all_pred.extend(pred_per_cat[category]['pred'])
        all_ans.extend(pred_per_cat[category]['ans'])

    acc_per_cat['total'] = accuracy_score(all_pred, all_ans)

    # Return test result.
    return acc_per_cat


@torch.no_grad()
def analogy_evaluation(
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        model: lmp.model.BaseRNNModel,
        tokenizer: lmp.tokenizer.BaseTokenizer
) -> None:
    r"""Helper function for test syntatic and semainic

    Args:
        device:
            Model running device.
        model:
            Language model.
        tokenizer:
            Tokenizer for get vocab_size, convert word id to word and contrast.
        data_loader:
            To get test data.
    """
    acc_per_cat = calc_correct_percent(
        data_loader=data_loader,
        device=device,
        model=model,
        tokenizer=tokenizer
    )
    for category in acc_per_cat:
        print(category, acc_per_cat[category])
