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

    config = lmp.config.BaseConfig.load_from_file(
        f'{model_path}/config.pickle')

    tokenizer = lmp.util.load_tokenizer(model_path, config.tokenizer_type)

    model = lmp.util.load_model(
        model_path, config, tokenizer, config.model_type)

    for generated_str in model.generator(tokenizer=tokenizer,
                                         begin_of_sentence='今天',
                                         beam_width=4,
                                         max_len=300):
        print(generated_str)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required arguments.
    parser.add_argument("--experiment_no", type=int, default=1,
                        required=True, help="using which experiment_no data")

    args = parser.parse_args()

    generate_sentences(args)
