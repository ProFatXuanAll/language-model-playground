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
    project_root = os.path.abspath(f'{os.path.abspath(__file__)}/..')
    data_path = f'{project_root}/data'
    save_path = f'{data_path}/{args.experiment_no}'
    state_path = f'{save_path}/checkpoint{args.checkpoint}.pt'

    config_save_path = f'{save_path}/config.pickle'
    tokenizer_save_path = f'{save_path}/tokenizer.pickle'

    config = lmp.util.load_config(
        args,
        file_path=config_save_path)

    tokenizer = lmp.util.load_tokenizer_by_config(
        config=config,
        checkpoint=args.checkpoint,
        file_path=tokenizer_save_path)

    print(tokenizer.vocab_size())

    model = lmp.util.load_saved_model(
        config=config,
        file_path=state_path,
        tokenizer=tokenizer
    )

    for generated_str in model.generator(tokenizer=tokenizer,
                                         begin_of_sentence='今天',
                                         beam_width=4,
                                         max_len=64):
        print(generated_str)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required arguments.
    parser.add_argument(
        "--experiment_no",
        type=int, default=1,
        required=True,
        help="using which experiment_no data"
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=1,
        required=True,
        help="using which experiment_no data"
    )

    args = parser.parse_args()

    generate_sentences(args)
