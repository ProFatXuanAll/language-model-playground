"""Test construction utilities for all language models.

Test target:
- :py:meth:`lmp.util.model.create`.
"""

import lmp.util.model
from lmp.model import LSTM1997, LSTM2000, ElmanNet
from lmp.tknzr import BaseTknzr


def test_create_elman_net(tknzr: BaseTknzr) -> None:
  """Test construction for :py:class:`lmp.model.ElmanNet`."""
  model = lmp.util.model.create(d_emb=10, model_name=ElmanNet.model_name, tknzr=tknzr)
  assert isinstance(model, ElmanNet)
  assert model.emb.embedding_dim == 10


def test_create_lstm_1997(tknzr: BaseTknzr) -> None:
  """Test construction for :py:class:`lmp.model.LSTM1997`."""
  model = lmp.util.model.create(d_cell=8, d_emb=10, model_name=LSTM1997.model_name, n_cell=4, tknzr=tknzr)
  assert isinstance(model, LSTM1997)
  assert model.d_cell == 8
  assert model.emb.embedding_dim == 10
  assert model.n_cell == 4


def test_create_lstm_2000(tknzr: BaseTknzr) -> None:
  """Test construction for :py:class:`lmp.model.LSTM2000`."""
  model = lmp.util.model.create(d_cell=8, d_emb=10, model_name=LSTM2000.model_name, n_cell=4, tknzr=tknzr)
  assert isinstance(model, LSTM2000)
  assert model.d_cell == 8
  assert model.emb.embedding_dim == 10
  assert model.n_cell == 4
