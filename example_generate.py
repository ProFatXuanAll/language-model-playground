import os
import pickle
import pandas as pd
import torch
import argparse

import lmp


#####################################################################
# 檔案路徑設定
#####################################################################


def generate_sentences(experiment_no=1):
    data_path = os.path.abspath('./data')
    model_path = f'{data_path}/{experiment_no}'

    config = lmp.config.BaseConfig.load_from_file(f'{model_path}/config.pickle')
    tokenizer = lmp.tokenizer.CharTokenizerByList.load_from_file(f'{model_path}/tokenizer.pickle')


    model = lmp.model.LSTMModel(config=config,
                            tokenizer=tokenizer)
    model.load_state_dict(torch.load(f'{model_path}/model.ckpt'))




    for generated_str in model.generator(tokenizer=tokenizer,
                                        begin_of_sentence='今天' ,
                                        beam_width=4,
                                        max_len=300):
        print(generated_str)
        print()



parser = argparse.ArgumentParser()
parser.add_argument("--experiment_no", type=int, default=1,  required=True, help="using which experiment_no data")
args = parser.parse_args()


generate_sentences(args.experiment_no)
