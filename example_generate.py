import os
import pickle
import pandas as pd
import torch

import lmp

#####################################################################
# 檔案路徑設定
#####################################################################
experiment_no = 1
data_path = os.path.abspath('./data')
model_path = f'{data_path}/{experiment_no}'

config = lmp.config.BaseConfig.load_from_file(f'{model_path}/config.pickle')

tokenizer = lmp.tokenizer.CharTokenizer.load_from_file(f'{model_path}/tokenizer.pickle')

model = lmp.model.GRUModel(config=config,
                           tokenizer=tokenizer)
model.load_state_dict(torch.load(f'{model_path}/model.ckpt'))

for generated_str in model.generator(tokenizer=tokenizer,
                                     begin_of_sentence='今天',
                                     beam_width=4,
                                     max_len=200):
    print(generated_str)
    print()