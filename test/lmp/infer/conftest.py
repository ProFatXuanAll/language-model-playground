"""Setup fixtures for testing :py:mod:`lmp.infer.`."""

import pytest
import torch

from lmp.model import BaseModel, RNNModel
from lmp.tknzr import BaseTknzr, CharTknzr


@pytest.fixture
def tknzr() -> BaseTknzr:
  """Example tokenizer instance."""

  return CharTknzr(
    is_uncased=True,
    max_vocab=-1,
    min_count=1,
    tk2id={
      CharTknzr.bos_tk: CharTknzr.bos_tkid,
      CharTknzr.eos_tk: CharTknzr.eos_tkid,
      CharTknzr.pad_tk: CharTknzr.pad_tkid,
      CharTknzr.unk_tk: CharTknzr.unk_tkid,
      'a': 4,
      'b': 5,
      'c': 6,
    },
  )


@pytest.fixture
def model(tknzr: BaseTknzr) -> BaseModel:
  """Example language model instance."""

  class ExampleModel(RNNModel):
    """Dummy model.

        Only used in inference testing.
        Designed to always predict token with largest token id in tokenizer's
        vocabulary.
        Thus only implement :py:meth:`lmp.model.BaseModel.pred` method.
        """

    def pred(self, batch_prev_tkids: torch.Tensor) -> torch.Tensor:
      """Predict largest token id in tokenizer's vocabulary."""
      batch_size = batch_prev_tkids.shape[0]
      seq_len = batch_prev_tkids.shape[1]

      # Output shape: (B, S, V).
      out = []

      for _ in range(batch_size):
        seq_tmp = []
        for _ in range(seq_len):
          vocab_tmp = []
          for _ in range(tknzr.vocab_size - 1):
            vocab_tmp.append(0.0)

          # Always predict largest token id in tokenizer's vocabuary.
          vocab_tmp.append(1.0)
          seq_tmp.append(vocab_tmp)
        out.append(seq_tmp)

      return torch.Tensor(out)

  return ExampleModel(
    d_emb=1,
    d_hid=1,
    n_hid_lyr=1,
    n_post_hid_lyr=1,
    n_pre_hid_lyr=1,
    p_emb=0.0,
    p_hid=0.0,
    tknzr=tknzr,
  )
