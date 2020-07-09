import os
import pickle
import pandas as pd
import torch
import sys

import lmp


#####################################################################
# 檔案路徑設定
#####################################################################

# experiment_no = 1
experiment_no = 5
data_path = os.path.abspath('./data')
model_path = f'{data_path}/{experiment_no}'

config = lmp.config.BaseConfig.load_from_file(f'{model_path}/config.pickle')
tokenizer = lmp.tokenizer.CharTokenizer.load_from_file(f'{model_path}/tokenizer.pickle')

# tokenizer = lmp.tokenizer.CharTokenizerDict.load_from_file(f'{model_path}/tokenizer.pickle')

# model = lmp.model.GRUModel(config=config,
#                            tokenizer=tokenizer)
model = lmp.model.LSTMModel(config=config,
                           tokenizer=tokenizer)
model.load_state_dict(torch.load(f'{model_path}/model.ckpt'))




BOS = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != ''  else '今天'

for generated_str in model.generator(tokenizer=tokenizer,
                                     begin_of_sentence=BOS ,
                                     beam_width=4,
                                     max_len=300):
    print(generated_str)
    print()
