"""Test construction utilities for all language models.

Test target:
- :py:meth:`lmp.util.model.create`.
"""

import lmp.util.model
from lmp.model import ElmanNet
from lmp.tknzr import BaseTknzr


def test_create_elman_net(tknzr: BaseTknzr) -> None:
  """Test construction for :py:class:`lmp.model.ElmanNet`."""
  model = lmp.util.model.create(d_emb=10, model_name=ElmanNet.model_name, tknzr=tknzr)
  assert isinstance(model, ElmanNet)
  assert model.emb.embedding_dim == 10
