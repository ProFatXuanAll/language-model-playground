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
def predict_word(
        test_data: List[str],
        device: torch.device,
        model: lmp.model.BaseRNNModel,
        tokenizer: lmp.tokenizer.BaseTokenizer
) -> str:
    """ To predict word that `test_data[1]-test_data[0]+test_data[3]`
    represent.

    Args:
        model:
            Language model.
        test_data:
            To get test data.
        device:
            Compute running device.
        tokenizer:
            To convert word id to word or word to word id.
    """

    # (E, V)
    emb = model.embedding_layer.weight.transpose(0, 1)

    test_data = tokenizer.convert_tokens_to_ids(test_data)
    test_data = torch.tensor(test_data)

    # calc output_word's embed.
    out = (
        model.embedding_layer(test_data[1].to(device)) -
        model.embedding_layer(test_data[0].to(device)) +
        model.embedding_layer(test_data[2].to(device))
    )

    # (E, 1)
    out = out.unsqueeze(-1)

    # (V)
    pred = torch.nn.functional.cosine_similarity(out, emb, dim=0)

    # (1)
    pred = pred.argmax(dim=-1).to('cpu').tolist()

    # Convert back to sequence.
    output_word = tokenizer.convert_id_to_token(pred)

    return output_word


@torch.no_grad()
def analogy_inference(
        test_data: List[str],
        device: torch.device,
        model: lmp.model.BaseRNNModel,
        tokenizer: lmp.tokenizer.BaseTokenizer
) -> str:
    r"""Helper function for analogy inference.

    Args:
        device:
            Model running device.
        model:
            Language model.
        tokenizer:
            Tokenizer for get vocab_size, convert word id to word and contrast.
        test_data:
            To get test data.
    """
    result = predict_word(
        test_data=test_data,
        device=device,
        model=model,
        tokenizer=tokenizer
    )
    print(
        test_data[0],
        ' : ',
        test_data[1],
        ' = ',
        test_data[2],
        ' : ',
        result)
    return result
