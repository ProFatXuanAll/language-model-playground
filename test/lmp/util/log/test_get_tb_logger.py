"""Test tensorboard logger.

Test target:
- :py:meth:`lmp.util.log.get_tb_logger`.
"""

import os

import torch.utils.tensorboard

import lmp.util.log


def test_get_tb_logger(exp_name: str, log_dir_path: str) -> None:
  """Must return tensorboard instance."""
  writer = lmp.util.log.get_tb_logger(exp_name=exp_name)
  assert isinstance(writer, torch.utils.tensorboard.SummaryWriter)

  writer.add_scalar('test', 123)
  writer.flush()
  writer.close()

  assert os.path.exists(log_dir_path)
