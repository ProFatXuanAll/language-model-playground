r"""Helper function for loading dataset.

Usage:
    dataset = lmp.util.load_dataset()
"""
import torch
import os
import pandas as pd
from typing import Union

import lmp.dataset


def load_dataset(data_path: str):
    # Load data.
    df = pd.read_csv(f'{data_path}/news_collection.csv')

    return lmp.dataset.BaseDataset(text_list=df['title'])
