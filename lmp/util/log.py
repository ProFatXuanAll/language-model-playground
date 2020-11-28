r"""Logging utilities."""

import os

import torch.utils.tensorboard

import lmp


def get_tb_logger(exp_name: str) -> torch.utils.tensorboard.SummaryWriter:
    r"""Get tensorboard logger.

    Create tensorboard for performance log visualization.
    Logs will be written to ``log/exp_name``.

    Parameters
    ==========
    exp_name: str
        Name of the logging experiment.

    Returns
    =======
    torch.utils.tensorboard.SummaryWriter
        Tensorboard logger instance.
    """
    file_dir = os.path.join(lmp.path.LOG_PATH, exp_name)

    if not os.path.exists:
        os.makedirs(file_dir)
    elif os.path.isfile(file_dir):
        raise FileExistsError(f'{file_dir} is a file.')

    return torch.utils.tensorboard.SummaryWriter(log_dir=file_dir)
