"""Test loading utilities for all language models.

Test target:
- :py:meth:`lmp.util.model.load`.
"""

import os

import torch

import lmp.util.model
from lmp.model import BaseModel, ElmanNet
from lmp.tknzr import BaseTknzr


def test_save_file(ckpt_dir_path: str, exp_name: str, model: BaseModel) -> None:
  """Must save checkpoints to correct path."""
  lmp.util.model.save(ckpt=0, exp_name=exp_name, model=model)
  assert os.path.exists(os.path.join(ckpt_dir_path, 'model-0.pt'))
  lmp.util.model.save(ckpt=1, exp_name=exp_name, model=model)
  assert os.path.exists(os.path.join(ckpt_dir_path, 'model-1.pt'))


def test_save_and_load(ckpt_dir_path: str, exp_name: str, model: BaseModel) -> None:
  """Ensure save and load consistency."""
  lmp.util.model.save(ckpt=0, exp_name=exp_name, model=model)
  load_model = lmp.util.model.load(ckpt=0, exp_name=exp_name)
  for (p_1, p_2) in zip(load_model.parameters(), model.parameters()):
    assert torch.equal(p_1, p_2)


def test_load_last(ckpt_dir_path: str, exp_name: str, model: BaseModel, tknzr: BaseTknzr) -> None:
  """Load the last checkpoint when ``ckpt == -1``."""
  last_model = ElmanNet(d_emb=10, tknzr=tknzr)
  lmp.util.model.save(ckpt=0, exp_name=exp_name, model=model)
  lmp.util.model.save(ckpt=1, exp_name=exp_name, model=last_model)
  load_model = lmp.util.model.load(ckpt=1, exp_name=exp_name)
  for (p_1, p_2) in zip(load_model.parameters(), last_model.parameters()):
    assert torch.equal(p_1, p_2)
