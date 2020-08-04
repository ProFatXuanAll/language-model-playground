# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re

from typing import List

# 3rd-party modules

import torch

# self-made modules

import lmp


@torch.no_grad()
def get_embedding_table(
    device: torch.device,
    model: lmp.model.BaseRNNModel,
    tokenizer: lmp.tokenizer.BaseTokenizer
) -> torch.tensor:
    r""" To Get a table that map word's id into embedding form.

    Args:
        model:
            Language model.
        tokenizer:
            Tokenizer for get vocab_size.
        device:
            Model running device.
    """
    vocab_size=tokenizer.vocab_size

    id_list=[i for i in range(vocab_size)]

    model.eval()

    inputs=torch.LongTensor(id_list).to(device)
    embed_words=model.get_word_embed(inputs)

    return embed_words

def calc_correct_percent(
    embed_table: torch.tensor,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    tokenizer: lmp.tokenizer.BaseTokenizer,
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

    # Initial variable.
    total_test_num=len(data_loader.dataset)
    TokenToId=tokenizer.convert_token_to_id
    IdToToken=tokenizer.convert_id_to_token

    # Record number of correct test.
    correct_num=0

    # Record number of processed data.
    processed_data=0

    # Syntatic and semaintic test.
    for word_a, word_b, word_c, word_d in data_loader:
        # Get batch size.
        batch_size=len(word_a)

        # Add precessed data.
        processed_data+=batch_size

        # Create a [batch_size, vocab_size, embed_dim] temp_embed_table.
        temp_embed_table=embed_table
        temp_embed_table=torch.unsqueeze(temp_embed_table, dim=0)
        temp_embed_table=temp_embed_table.repeat(batch_size, 1, 1)

        # Convert word id to word vector.
        embed_a=torch.stack([embed_table[TokenToId(word)] for word in word_a], dim=0)
        embed_b=torch.stack([embed_table[TokenToId(word)] for word in word_b], dim=0)
        embed_c=torch.stack([embed_table[TokenToId(word)] for word in word_c], dim=0)

        # Calculate predict word vector.
        predict_embed_d=(embed_b-embed_a+embed_c)

        # Transform the size of predict_embed_d from [batch_size, embed_dim] to [batch_size, 1, embed_dim]
        predict_embed_d=torch.unsqueeze(predict_embed_d, dim=1)
        # Duplicate dimension1 to vocab size.
        # Result size [batch_size, vocab_size, embed_dim].
        predict_embed_d=predict_embed_d.repeat(1, tokenizer.vocab_size, 1)

        # Calculate similarity.
        predict_embed_d=predict_embed_d.to(device)
        similaritys=torch.cosine_similarity(predict_embed_d, temp_embed_table, dim=2)

        # Check the answer if correct.
        similaritys=similaritys.tolist()
        for i in range(batch_size):
            # print(IdToToken(similaritys[i].index(max(similaritys[i]))), ' ', word_d[i])
            predict_word=IdToToken(similaritys[i].index(max(similaritys[i])))

            # If answer is correct then correct_num add 1.
            if  predict_word == word_d[i]:
                correct_num+=1

        # Print Execution progress.
        print(processed_data, '\\', total_test_num)

    # Return test result.
    return correct_num/total_test_num







@torch.no_grad()
def syntatic_and_semantic_test(
        device:torch.device,
        model: lmp.model.BaseRNNModel,
        data_loader: torch.utils.data.DataLoader,
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
    embed_table=get_embedding_table(
        device,
        model,
        tokenizer
        )
    result=calc_correct_percent(
        embed_table,
        data_loader,
        device,
        tokenizer
    )
    print('score:', result)