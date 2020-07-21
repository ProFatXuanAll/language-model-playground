r"""Helper function for loading model.

Usage:
    model = lmp.util.load_saved_model()
"""
import torch
from typing import Union

import lmp.config
import lmp.tokenizer
import lmp.model


def load_saved_model(
    file_path: str,
    config: lmp.config.BaseConfig,
    tokenizer: Union[lmp.tokenizer.BaseTokenizerByList,
                     lmp.tokenizer.BaseTokenizerByDict]
) -> Union[lmp.model.GRUModel, lmp.model.LSTMModel]:
    r"""Used to load saved model

    Args:
        config:
            Configuration of model.
            Come from lmp.config.BaseConfig.
        tokenizer:
            Convert sentences to ids, and decode the result ids to sentences.
    Returns:
        lmp.model.GRUModel
        lmp.model.LSTMModel
    """

    if config.model_class.lower() not in ['lstm', 'gru']:
        raise ValueError(
            f'model `{config.model_class}` is not exist, please input lstm or gru')
    if config.model_class.lower() == 'gru':
        model = lmp.model.GRUModel(config=config, tokenizer=tokenizer)
    elif config.model_class.lower() == 'lstm':
        model = lmp.model.LSTMModel(config=config, tokenizer=tokenizer)

    checkpoint_state = torch.load(file_path)
    model.load_state_dict(checkpoint_state['model'])

    return model


def load_blank_model(
    config: lmp.config.BaseConfig,
    tokenizer: Union[lmp.tokenizer.BaseTokenizerByList,
                     lmp.tokenizer.BaseTokenizerByDict]
) -> Union[lmp.model.GRUModel, lmp.model.LSTMModel]:
    r"""Used to load blank model

    Args:
        config:
            Configuration of model.
            Come from lmp.config.BaseConfig.
        tokenizer:
            Convert sentences to ids, and decode the result ids to sentences.
    Returns:
        lmp.model.GRUModel
        lmp.model.LSTMModel
    """

    if config.model_class.lower() not in ['lstm', 'gru']:
        raise ValueError(
            f'model `{config.model_class}` is not exist, please input lstm or gru')
    if config.model_class.lower() == 'gru':
        model = lmp.model.GRUModel(config=config, tokenizer=tokenizer)
    elif config.model_class.lower() == 'lstm':
        model = lmp.model.LSTMModel(config=config, tokenizer=tokenizer)

    return model


def load_model_for_train(
    checkpoint: int,
    config: lmp.config.BaseConfig,
    device: torch.device,
    save_path: str,
    tokenizer: Union[lmp.tokenizer.BaseTokenizerByList,
                     lmp.tokenizer.BaseTokenizerByDict]

) -> Union[lmp.model.GRUModel, lmp.model.LSTMModel]:

    if checkpoint > 0:
        state_path = f'{save_path}/checkpoint{checkpoint}.pt'
        model = load_saved_model(file_path=state_path,
                                 config=config, tokenizer=tokenizer)
    else:
        model = load_blank_model(config=config, tokenizer=tokenizer)

    model.to(device)

    return model
