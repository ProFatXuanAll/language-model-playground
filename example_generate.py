import os
import pickle
import pandas as pd
import torch

import char_rnn

#####################################################################
# 檔案路徑設定
#####################################################################
data_path = 'data'

train_file = '{}/old-newspaper.tsv'.format(data_path)
train_converter_file = '{}/train.converter.pickle'.format(data_path)
train_model_file = '{}/train.model.ckpt'.format(data_path)

if os.path.exists(train_converter_file):
    converter = char_rnn.token.Converter()
    converter.load_from_file(train_converter_file)
else:
    raise FileNotFoundError('pretrained converter file {} does not exist.'.format(train_converter_file))

EMBED_DIM = 100
HIDDEN_DIM = 100

model = char_rnn.model.CharRNN(vocab_size=converter.vocab_size(),
                               embed_dim=EMBED_DIM,
                               hidden_dim=HIDDEN_DIM,
                               pad_token_id=converter.pad_token_id)

if os.path.exists(train_model_file):
    model.load_state_dict(torch.load(train_model_file))
else:
    raise FileNotFoundError('pretrained model file {} does not exist.'.format(train_model_file))

for generate_str in char_rnn.model.generator(model=model,
                                             converter=converter,
                                             begin_of_sentence='高宏宇',
                                             beam_width=4,
                                             max_len=500):
    print(generate_str)
    print()