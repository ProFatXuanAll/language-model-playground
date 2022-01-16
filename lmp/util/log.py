"""Logging utilities."""

import os

import tensorboardX

import lmp


def get_tb_logger(exp_name: str) -> tensorboardX.SummaryWriter:
  """Get tensorboard logger.

  Create tensorboard for performance log visualization.
  Logs will be written to ``log/exp_name``.

  Parameters
  ----------
  exp_name: str
    Name of the logging experiment.

  Returns
  -------
  tensorboardX.SummaryWriter
    Tensorboard logger instance.
  """
  file_dir = os.path.join(lmp.util.path.LOG_PATH, exp_name)

  if not os.path.exists:
    os.makedirs(file_dir)
  elif os.path.isfile(file_dir):
    raise FileExistsError(f'{file_dir} is a file.')

  return tensorboardX.SummaryWriter(log_dir=file_dir)
