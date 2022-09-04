"""Logging utilities."""

import os

# Typeshed for `tensorboardX` is not available, we ignore type check on `tensorboardX`.
import tensorboardX  # type: ignore

import lmp.util.validate
import lmp.vars


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
  # `exp_name` validation.
  lmp.util.validate.raise_if_not_instance(val=exp_name, val_name='exp_name', val_type=str)
  lmp.util.validate.raise_if_empty_str(val=exp_name, val_name='exp_name')

  # `log_dir` validation.
  log_dir = os.path.join(lmp.vars.LOG_PATH, exp_name)
  lmp.util.validate.raise_if_is_file(path=log_dir)

  if not os.path.exists(log_dir):
    os.makedirs(log_dir)

  return tensorboardX.SummaryWriter(log_dir=log_dir)
