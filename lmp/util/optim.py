"""Optimization utilities."""

import torch

import lmp.util.validate
from lmp.model import BaseModel


def get_optimizer(
  beta1: float,
  beta2: float,
  eps: float,
  lr: float,
  model: BaseModel,
  wd: float,
) -> torch.optim.AdamW:
  """Get AdamW optimizer.

  Parameters
  ----------
  beta1: float
    First coefficient of gradient moving average.
  beta2: float
    Second coefficient of gradient moving average.
  eps: float
    Numerically saved computation term.
  lr: float
    Learning rate of gradient descent.
  model: lmp.model.BaseModel
    Language model to be optimized.
  wd: float
    Weight decay coefficient.

  Returns
  -------
  torch.optim.AdamW
    Language model optimizer.

  See Also
  --------
  torch.optim.AdamW
    AdamW algorithm.
  """
  # `beta1` validation.
  lmp.util.validate.raise_if_not_instance(val=beta1, val_name='beta1', val_type=float)
  lmp.util.validate.raise_if_wrong_ordered(vals=[0.0, beta1, 1.0], val_names=['0.0', 'beta1', '1.0'])

  # `beta2` validation.
  lmp.util.validate.raise_if_not_instance(val=beta2, val_name='beta2', val_type=float)
  lmp.util.validate.raise_if_wrong_ordered(vals=[0.0, beta2, 1.0], val_names=['0.0', 'beta2', '1.0'])

  # `eps` validation.
  lmp.util.validate.raise_if_not_instance(val=eps, val_name='eps', val_type=float)
  lmp.util.validate.raise_if_wrong_ordered(vals=[0.0, eps], val_names=['0.0', 'eps'])

  # `lr` validation.
  lmp.util.validate.raise_if_not_instance(val=lr, val_name='lr', val_type=float)
  lmp.util.validate.raise_if_wrong_ordered(vals=[0.0, lr], val_names=['0.0', 'lr'])

  # `model` validation.
  lmp.util.validate.raise_if_not_instance(val=model, val_name='model', val_type=BaseModel)

  # `wd` validation.
  lmp.util.validate.raise_if_not_instance(val=wd, val_name='wd', val_type=float)
  lmp.util.validate.raise_if_wrong_ordered(vals=[0.0, wd], val_names=['0.0', 'wd'])

  # Remove weight decay on bias and layer-norm.  This can only be done after moving model to running device.
  no_decay = ['bias', 'LayerNorm.weight']
  optim_group_params = [
    {
      'params': [param for name, param in model.named_parameters() if not any(nd in name for nd in no_decay)],
      'weight_decay': wd,
    },
    {
      'params': [param for name, param in model.named_parameters() if any(nd in name for nd in no_decay)],
      'weight_decay': 0.0,
    },
  ]

  # Get new optimizer instance.
  return torch.optim.AdamW(optim_group_params, betas=(beta1, beta2), eps=eps, lr=lr)


def get_scheduler(optim: torch.optim.AdamW, total_step: int, warmup_step: int) -> torch.optim.lr_scheduler.LambdaLR:
  """Get linearly decay scheduler with linearly warm up.

  Learning rate will first linearly increase (warm up) to the specified value, then linearly decay to ``0``.

  Parameters
  ----------
  optim: torch.optim.AdamW
    Optimizer to be scheduled.
  total_step: int
    Total training step.
  warmup_step: int
    Learning rate warmup step.

  Returns
  -------
  torch.optim.lr_scheduler.LambdaLR
    Optimizer learning rate scheduler.
  """
  # `optim` validation.
  lmp.util.validate.raise_if_not_instance(val=optim, val_name='optim', val_type=torch.optim.AdamW)

  def lr_lambda(step: int) -> float:
    # Warm up phase.
    if step < warmup_step:
      return float(step / max(1, warmup_step))

    # Decay phase.
    return float(max(0, (total_step - step) / max(1, total_step - warmup_step)))

  return torch.optim.lr_scheduler.LambdaLR(optimizer=optim, lr_lambda=lr_lambda, last_epoch=-1)
