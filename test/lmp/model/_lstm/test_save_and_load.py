"""Test save and load operation for model parameters.

Test target:
- :py:meth:`lmp.model.LSTMModel.load`.
- :py:meth:`lmp.model.LSTMModel.save`.
"""

import torch

from lmp.model import LSTMModel
from lmp.tknzr import BaseTknzr


def test_save_and_load(tknzr: BaseTknzr, ckpt: int, exp_name: str, clean_model):
  """Saved parameters are the same as loaded."""
  model = LSTMModel(
    d_emb=1,
    d_hid=1,
    n_hid_lyr=1,
    n_pre_hid_lyr=1,
    n_post_hid_lyr=1,
    p_emb=0.5,
    p_hid=0.5,
    tknzr=tknzr,
  )

  # Save model parameters.
  model.save(
    ckpt=ckpt,
    exp_name=exp_name,
  )

  # Load model parameters.
  load_model = LSTMModel.load(
    ckpt=ckpt,
    exp_name=exp_name,
    d_emb=1,
    d_hid=1,
    n_hid_lyr=1,
    n_pre_hid_lyr=1,
    n_post_hid_lyr=1,
    p_emb=0.5,
    p_hid=0.5,
    tknzr=tknzr,
  )

  # Ensure parameters are the same.
  for p_1, p_2 in zip(model.parameters(), load_model.parameters()):
    assert torch.equal(p_1, p_2)
