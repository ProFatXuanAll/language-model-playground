r"""Helper function for loading model.

Usage:
    model = lmp.util.load_model()
"""
import torch
from typing import Union

import lmp.config
import lmp.tokenizer


def load_model(model_path: str,
               config: lmp.config.BaseConfig,
               tokenizer: Union[lmp.tokenizer.BaseTokenizerByList, lmp.tokenizer.BaseTokenizerByDict],
               model_type: str = 'lstm') -> Union[lmp.model.GRUModel, lmp.model.LSTMModel]:
    r"""Used to load saved model

    Args:
        config:
            Configuration of model.
            Come from lmp.config.BaseConfig.
        tokenizer:
            Convert sentences to ids, and decode the result ids to sentences.
        model_type:
            Decide to use which model, LSTM or GRU.
    Returns:
        lmp.model.GRUModel
        lmp.model.LSTMModel
    """

    if config.model_type.lower() not in ['lstm', 'gru']:
        raise ValueError(
            f'model `{args.model}` is not exist, please input lstm or gru')
    if model_type.lower() == 'gru':
        model = lmp.model.GRUModel(config=config, tokenizer=tokenizer)
    elif model_type.lower() == 'lstm':
        model = lmp.model.LSTMModel(config=config, tokenizer=tokenizer)

    model.load_state_dict(torch.load(f'{model_path}/model.ckpt'))

    return model


def load_blank_model(config: lmp.config.BaseConfig,
                     tokenizer: Union[lmp.tokenizer.BaseTokenizerByList, lmp.tokenizer.BaseTokenizerByDict],
                     model_type: str = 'lstm') -> Union[lmp.model.GRUModel, lmp.model.LSTMModel]:
    r"""Used to load blank model

    Args:
        config:
            Configuration of model.
            Come from lmp.config.BaseConfig.
        tokenizer:
            Convert sentences to ids, and decode the result ids to sentences.
        model_type:
            Decide to use which model, LSTM or GRU.
    Returns:
        lmp.model.GRUModel
        lmp.model.LSTMModel
    """

    if config.model_type.lower() not in ['lstm', 'gru']:
        raise ValueError(
            f'model `{args.model}` is not exist, please input lstm or gru')
    if model_type.lower() == 'gru':
        model = lmp.model.GRUModel(config=config, tokenizer=tokenizer)
    elif model_type.lower() == 'lstm':
        model = lmp.model.LSTMModel(config=config, tokenizer=tokenizer)

    return model
