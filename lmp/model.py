r"""text generation models.

Usage:
    model = lmp.model.LSTMModel(...)
    model = lmp.model.GRUModel(...)
"""
# built-in modules
import os

# 3rd-party modules
import numpy as np
import pandas as pd
import torch
import torch.nn
import torch.utils.data
import torch.nn.utils.rnn
import torch.optim
import sklearn.metrics
from tqdm import tqdm
from typing import List, Union
import lmp.tokenizer
import lmp.config
##############################################
# build model
##############################################


class BaseModel(torch.nn.Module):
    r"""Text-generation's BaseModel.

    Args:
        config:
            Configuration for training model.
            Come from lmp.config.BaseConfig.
        tokenizer:
            Convert sentences to ids, and decode the result ids to sentences.

    """
    def __init__(self, config: lmp.config.BaseConfig,
                 tokenizer: Union[lmp.tokenizer.BaseTokenizerByList, lmp.tokenizer.BaseTokenizerByDict]):
        super(BaseModel, self).__init__()
        
        # Embedding layer
        self.embedding_layer = torch.nn.Embedding(num_embeddings=tokenizer.vocab_size(),
                                                  embedding_dim=config.embedding_dim,
                                                  padding_idx=tokenizer.pad_token_id)
        

        # RNN layer
        self.rnn_layer = torch.nn.RNN(input_size=config.embedding_dim,
                                      hidden_size=config.hidden_dim,
                                      num_layers=config.num_rnn_layers,
                                      dropout=config.dropout,
                                      batch_first=True)
        
        # Forward passing
        # self.linear = torch.nn.Linear(config.hidden_dim, config.embedding_dim)
        self.linear = []

        for _ in range(config.num_linear_layers):
            self.linear.append(torch.nn.Linear(
                config.hidden_dim, config.hidden_dim))
            self.linear.append(torch.nn.ReLU())
            self.linear.append(torch.nn.Dropout(config.dropout))

        self.linear.append(torch.nn.Linear(
            config.hidden_dim, config.embedding_dim))
        
        self.sequential = torch.nn.Sequential(*self.linear)

    def forward(self, batch_x: torch.Tensor) -> torch.Tensor:
        ######################################################################
        # embedding前 的 batch_x 維度: (batch_Size, sequence_length)
        # 
        # embedding後 的 batch_x維度: (batch_size, sequence_length, embedding_dimension)
        ######################################################################
        batch_x = self.embedding_layer(batch_x)
  
        ######################################################################
        # ht 維度: (batch_size, sequence_length, hidden_dimension)
        ######################################################################
        ht, _ = self.rnn_layer(batch_x)

        
        ######################################################################
        # ht 維度: (batch_size, sequence_length, embedding_dimension)
        ######################################################################
        #ht = self.linear(ht)
        ht = self.sequential(ht)

        ######################################################################
        # yt 維度: (batch_size, sequence_length, vocabulary_size)
        ######################################################################
        yt = ht.matmul(self.embedding_layer.weight.transpose(0, 1))
        
        return yt

    def generator(self,
                  tokenizer: Union[lmp.tokenizer.BaseTokenizerByList, lmp.tokenizer.BaseTokenizerByDict],
                  begin_of_sentence: str = '',
                  beam_width: int = 4,
                  max_len: int = 200) -> List[str]:
        r"""Using beam search algorithm to generate texts.

        Args:
             tokenizer:
                Convert sentences to ids, and decode the result ids to sentences.
            begin_of_sentence:
                As input of model to find probability of next word.
            beam_width:
                Used for Beam search algorithm to find 'beam_width' candidates.  
            max_len:
                Maximum of output sentence's length.
        Returns:
            generating sentences by using beam search algorithm.
        """
        if begin_of_sentence is None or len(begin_of_sentence) == 0:
            raise ValueError('`begin_of_sentence` should be list type object.')

        generate_result: List[List[int]] = []

        with torch.no_grad():
            all_ids = tokenizer.convert_sentences_to_ids([begin_of_sentence])
            all_ids_prob = [0]
            

            while True:
                active_ids = []
                active_ids_prob = []

                # 決定當前還有哪些句子需要進行生成
                for i in range(len(all_ids)):
                    if all_ids[i][-1] != tokenizer.eos_token_id and len(all_ids[i]) < max_len:
                        active_ids.append(all_ids[i])
                        active_ids_prob.append(all_ids_prob[i])
                    elif len(generate_result) < beam_width:
                        generate_result.append(all_ids[i])

                # 如果沒有需要生成的句子就結束迴圈
                if not active_ids or len(generate_result) >= beam_width:
                    break

                batch_x = [torch.LongTensor(ids) for ids in active_ids]
                
                batch_x = torch.nn.utils.rnn.pad_sequence(batch_x,
                                                          batch_first=True,
                                                          padding_value=tokenizer.pad_token_id)
                batch_pred_y = self(batch_x)

                
                # 從各個句子中的 beam 挑出前 beam_width 個最大值
                top_value = {}
                for beam_id in range(len(active_ids)):
                    current_beam_vocab_pred = batch_pred_y[beam_id][len(
                        active_ids[beam_id]) - 1]   # current_beam_vocab_pred.shape() = ([vocab_size])
                    current_beam_vocab_pred = torch.nn.functional.softmax(
                        current_beam_vocab_pred, dim=0) 
                    current_beam_top_value = [{
                        'vocab_id': tokenizer.eos_token_id,
                        'value': 0
                    } for i in range(beam_width)]

                    
                    for vocab_id in range(tokenizer.vocab_size()):
                        if current_beam_vocab_pred[vocab_id] < current_beam_top_value[0]['value']:
                            continue

                        for level in range(beam_width):
                            if current_beam_vocab_pred[vocab_id] < current_beam_top_value[level]['value']:
                                level -= 1
                                break

                        for tmp in range(level):
                            current_beam_top_value[tmp] = current_beam_top_value[tmp + 1]

                        current_beam_top_value[level] = {'vocab_id': vocab_id,
                                                         'value': current_beam_vocab_pred[vocab_id]}

                    top_value[beam_id] = current_beam_top_value
                
                # 從 beam_width ** 2 中挑出 beam_width 個最大值
                final_top_value = []

                for i in range(beam_width):
                    max_value_beam_id = 0
                    max_value_vocab_id = tokenizer.eos_token_id
                    min_value = 999999999

                    for beam_id in range(len(top_value)):
                        value = - \
                            torch.log(
                                top_value[beam_id][-1]['value']) + active_ids_prob[beam_id]
                        if value < min_value:
                            max_value_beam_id = beam_id
                            min_value = value.item()   # 從 tensor 變為 int，因為 min_value 是 int (不轉的話一樣可運行，但用 mypy 檢查會報 error )
                            max_value_vocab_id = top_value[beam_id][-1]['vocab_id']

                    final_top_value.append({
                        'beam_id': max_value_beam_id,
                        'vocab_id': max_value_vocab_id,
                        'value': min_value
                    })

                    top_value[max_value_beam_id].pop()

                # back to all_ids
                all_ids = []
                all_ids_prob = []
                for obj in final_top_value:
                    all_ids.append(
                        active_ids[obj['beam_id']] + [obj['vocab_id']])
                    all_ids_prob.append(obj['value'])

        for ids in generate_result:
            if ids[-1] == tokenizer.eos_token_id:
                ids.pop()
        return tokenizer.convert_ids_to_sentences(generate_result)


class LSTMModel(BaseModel):
    r"""LSTM replaces BaseModel's rnn_layer.
    """
    def __init__(self, config, tokenizer):
        super(LSTMModel, self).__init__(config, tokenizer)
        # rewrite RNN layer
        self.rnn_layer = torch.nn.LSTM(input_size=config.embedding_dim,
                                       hidden_size=config.hidden_dim,
                                       num_layers=config.num_rnn_layers,
                                       dropout=config.dropout,
                                       batch_first=True)


class GRUModel(BaseModel):
    r"""GRU replaces BaseModel's rnn_layer.
    """
    def __init__(self, config, tokenizer):
        super(GRUModel, self).__init__(config, tokenizer)

        # rewrite RNN layer
        self.rnn_layer = torch.nn.GRU(input_size=config.embedding_dim,
                                      hidden_size=config.hidden_dim,
                                      num_layers=config.num_rnn_layers,
                                      dropout=config.dropout,
                                      batch_first=True)
