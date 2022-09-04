"""Test get optimizer result.

Test target:
- :py:meth:`lmp.util.optim.get_optimizer`.
"""

import math

import torch

import lmp.util.optim
from lmp.model import BaseModel


def test_optimization_values(
  beta1: float,
  beta2: float,
  eps: float,
  lr: float,
  model: BaseModel,
  weight_decay: float,
) -> None:
  """Test construction for :py:class:`torch.optim.AdamW`."""
  optim = lmp.util.optim.get_optimizer(beta1=beta1, beta2=beta2, eps=eps, lr=lr, model=model, weight_decay=weight_decay)
  assert isinstance(optim, torch.optim.AdamW)
  assert math.isclose(optim.defaults['betas'][0], beta1)
  assert math.isclose(optim.defaults['betas'][1], beta2)
  assert math.isclose(optim.defaults['eps'], eps)
  assert math.isclose(optim.defaults['lr'], lr)
  assert math.isclose(optim.defaults['weight_decay'], weight_decay)
