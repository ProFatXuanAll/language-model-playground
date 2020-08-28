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
    pred = tokenizer.convert_id_to_token(pred)
    return pred


@torch.no_grad()
def analogy_eval(
    dataloader: lmp.dataset.AnalogyDataset,
    device: torch.device,
    model: Union[lmp.model.BaseRNNModel, lmp.model.BaseResRNNModel],
    tokenizer: lmp.tokenizer.BaseTokenizer
) -> Dict[str, float]:
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
    for category in acc_per_cat:
        print(category, acc_per_cat[category])
    return acc_per_cat
