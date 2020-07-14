r"""giving a text to generate sentences.
Usage:
    python example_generate.py --experiment_no 1 
--experiment_no is a required argument.
Run 'python example_generate.py --help' for help.
"""

import os
import pickle
import pandas as pd
import torch
import argparse

import lmp


#####################################################################
# 檔案路徑設定
#####################################################################


def generate_sentences(args):
    data_path = os.path.abspath('./data')
    model_path = f'{data_path}/{args.experiment_no}'

    config = lmp.config.BaseConfig.load_from_file(f'{model_path}/config.pickle')
    
    if config.tokenizer_type.lower() not in ['list', 'dict']:
        raise NameError(f'`{args.tokenizer}` is not exist, please input list or dict')
    if config.tokenizer_type.lower() == 'dict':
        tokenizer = lmp.tokenizer.CharTokenizerByDict().load_from_file(f'{model_path}/tokenizer.pickle')
    elif config.tokenizer_type.lower() == 'list':
        tokenizer = lmp.tokenizer.CharTokenizerByList().load_from_file(f'{model_path}/tokenizer.pickle')



    if config.model_type.lower() not  in ['lstm', 'gru']:
        raise NameError(f'model `{args.model}` is not exist, please input lstm or gru')
    if config.model_type.lower() == 'gru':
        model = lmp.model.GRUModel(config=config, tokenizer=tokenizer)
    elif config.model_type.lower() == 'lstm':
        model = lmp.model.LSTMModel(config=config, tokenizer=tokenizer)
        
    model.load_state_dict(torch.load(f'{model_path}/model.ckpt'))





    for generated_str in model.generator(tokenizer=tokenizer,
                                        begin_of_sentence='今天' ,
                                        beam_width=4,
                                        max_len=300):
        print(generated_str)
        print()



if  __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required arguments.
    parser.add_argument("--experiment_no", type=int, default=1,  required=True, help="using which experiment_no data")
  
    
    args = parser.parse_args()


    generate_sentences(args)
