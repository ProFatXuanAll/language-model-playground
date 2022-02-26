"""Test creation of tensor datasets.

Test target:
- :py:meth:`lmp.util.dset.FastTensorDset`.
- :py:meth:`lmp.util.dset.SlowTensorDset`.
"""

import torch

import lmp.util.dset
from lmp.dset import WikiText2Dset
from lmp.tknzr import WsTknzr


def test_fast_tensor_dset(max_seq_len: int) -> None:
  """Load dataset in memory and convert to tensor."""
  tknzr = WsTknzr(is_uncased=True, max_vocab=-1, min_count=10)
  tknzr.build_vocab(batch_txt=['a', 'b', 'c'])

  wiki_dset = WikiText2Dset(ver='valid')

  dset = lmp.util.dset.FastTensorDset(dset=wiki_dset, max_seq_len=max_seq_len, tknzr=tknzr)

  assert isinstance(dset, lmp.util.dset.FastTensorDset)
  assert len(dset) == len(wiki_dset)
  for idx, tkids in enumerate(dset):
    assert isinstance(tkids, torch.Tensor), 'Each sample in the tensor dataset must be tensor.'
    assert tkids.size() == torch.Size([max_seq_len]), 'Each sample in the tensor dataset must have same length.'
    assert torch.all(dset[idx] == tkids), 'Support ``__getitem__`` and ``__iter__``.'


def test_slow_tensor_dset(max_seq_len: int) -> None:
  """Load dataset and convert to tensor on the fly."""
  tknzr = WsTknzr(is_uncased=True, max_vocab=-1, min_count=10)
  tknzr.build_vocab(batch_txt=['a', 'b', 'c'])

  wiki_dset = WikiText2Dset(ver='valid')

  dset = lmp.util.dset.SlowTensorDset(dset=wiki_dset, max_seq_len=max_seq_len, tknzr=tknzr)

  assert isinstance(dset, lmp.util.dset.SlowTensorDset)
  assert len(dset) == len(wiki_dset)
  for idx, tkids in enumerate(dset):
    assert isinstance(tkids, torch.Tensor), 'Each sample in the tensor dataset must be tensor.'
    assert tkids.size() == torch.Size([max_seq_len]), 'Each sample in the tensor dataset must have same length.'
    assert torch.all(dset[idx] == tkids), 'Support ``__getitem__`` and ``__iter__``.'
