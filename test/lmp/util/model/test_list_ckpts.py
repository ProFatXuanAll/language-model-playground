"""Test model checkpoints listing utilities.

Test target:
- :py:meth:`lmp.util.model.list_ckpts`.
"""

import lmp.util.model
from lmp.model import BaseModel


def test_list_ckpts(ckpt_dir_path: str, exp_name: str, model: BaseModel) -> None:
  """List specified language model checkpoints."""
  lmp.util.model.save(ckpt=0, exp_name=exp_name, model=model)
  lmp.util.model.save(ckpt=1, exp_name=exp_name, model=model)
  lmp.util.model.save(ckpt=2, exp_name=exp_name, model=model)

  assert lmp.util.model.list_ckpts(exp_name=exp_name, first_ckpt=0, last_ckpt=2) == [0, 1, 2]
  assert lmp.util.model.list_ckpts(exp_name=exp_name, first_ckpt=-1, last_ckpt=2) == [2]
  assert lmp.util.model.list_ckpts(exp_name=exp_name, first_ckpt=-1, last_ckpt=0) == [2]
  assert lmp.util.model.list_ckpts(exp_name=exp_name, first_ckpt=0, last_ckpt=-1) == [0, 1, 2]
  assert lmp.util.model.list_ckpts(exp_name=exp_name, first_ckpt=1, last_ckpt=-1) == [1, 2]
