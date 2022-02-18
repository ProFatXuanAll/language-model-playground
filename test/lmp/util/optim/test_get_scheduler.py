"""Test get scheduler result.

Test target:
- :py:meth:`lmp.util.optim.get_scheduler`.
"""

import torch

import lmp.util.optim
from lmp.model import BaseModel


def test_scheduler(
  beta1: float,
  beta2: float,
  eps: float,
  lr: float,
  model: BaseModel,
  total_step: int,
  warmup_step: int,
  wd: float,
) -> None:
  """Test construction for :py:class:`torch.optim.lr_scheduler.LambdaLR`."""
  optim = lmp.util.optim.get_optimizer(beta1=beta1, beta2=beta2, eps=eps, lr=lr, model=model, wd=wd)
  schdl = lmp.util.optim.get_scheduler(optim=optim, total_step=total_step, warmup_step=warmup_step)
  assert isinstance(schdl, torch.optim.lr_scheduler.LambdaLR)
