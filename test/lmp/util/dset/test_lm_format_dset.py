"""Test language model formation.

Test target:
- :py:meth:`lmp.util.dset.LMFormatDset`.
"""

import torch

import lmp.tknzr
import lmp.util.dset
from lmp.dset import ChPoemDset, DemoDset, WikiText2Dset, WNLIDset


def test_formatting_chinese_poem(max_seq_len: int, stride: int) -> None:
  """Format Chinese poem datasets correctly."""
  for ver in ChPoemDset.vers:
    dset = ChPoemDset(ver=ver)

    tknzr = lmp.tknzr.CharTknzr()
    tknzr.build_vocab(batch_txt=dset)

    lm_format_dset = lmp.util.dset.LMFormatDset(
      dset=dset,
      max_seq_len=max_seq_len,
      stride=stride,
      tknzr=tknzr,
    )

    dset_size = len(lm_format_dset)
    assert isinstance(dset_size, int)

    for idx in range(dset_size):
      spl = lm_format_dset[idx]
      assert isinstance(spl, tuple)
      assert len(spl) == 2

      cur_tkids, next_tkids = spl
      assert isinstance(cur_tkids, torch.Tensor)
      assert cur_tkids.dtype == torch.long
      assert isinstance(next_tkids, torch.Tensor)
      assert next_tkids.dtype == torch.long
      assert torch.all(cur_tkids[1:] == next_tkids[:-1])

    assert hasattr(lm_format_dset, 'n_tk')
    assert isinstance(lm_format_dset.n_tk, int)
    assert lm_format_dset.n_tk > 0


def test_formatting_demo(max_seq_len: int, stride: int) -> None:
  """Format demo datasets correctly."""
  for ver in DemoDset.vers:
    dset = DemoDset(ver=ver)

    tknzr = lmp.tknzr.CharTknzr()
    tknzr.build_vocab(batch_txt=dset)

    lm_format_dset = lmp.util.dset.LMFormatDset(
      dset=dset,
      max_seq_len=max_seq_len,
      stride=stride,
      tknzr=tknzr,
    )

    dset_size = len(lm_format_dset)
    assert isinstance(dset_size, int)

    for idx in range(dset_size):
      spl = lm_format_dset[idx]
      assert isinstance(spl, tuple)
      assert len(spl) == 2

      cur_tkids, next_tkids = spl
      assert isinstance(cur_tkids, torch.Tensor)
      assert cur_tkids.dtype == torch.long
      assert isinstance(next_tkids, torch.Tensor)
      assert next_tkids.dtype == torch.long
      assert torch.all(cur_tkids[1:] == next_tkids[:-1])

    assert hasattr(lm_format_dset, 'n_tk')
    assert isinstance(lm_format_dset.n_tk, int)
    assert lm_format_dset.n_tk > 0


def test_formatting_wiki_text_2(max_seq_len: int, stride: int) -> None:
  """Format Wiki-text-2 datasets correctly."""
  for ver in WikiText2Dset.vers:
    dset = WikiText2Dset(ver=ver)

    tknzr = lmp.tknzr.WsTknzr()
    tknzr.build_vocab(batch_txt=dset)

    lm_format_dset = lmp.util.dset.LMFormatDset(
      dset=dset,
      max_seq_len=max_seq_len,
      stride=stride,
      tknzr=tknzr,
    )

    dset_size = len(lm_format_dset)
    assert isinstance(dset_size, int)

    for idx in range(dset_size):
      spl = lm_format_dset[idx]
      assert isinstance(spl, tuple)
      assert len(spl) == 2

      cur_tkids, next_tkids = spl
      assert isinstance(cur_tkids, torch.Tensor)
      assert cur_tkids.dtype == torch.long
      assert isinstance(next_tkids, torch.Tensor)
      assert next_tkids.dtype == torch.long
      assert torch.all(cur_tkids[1:] == next_tkids[:-1])

    assert hasattr(lm_format_dset, 'n_tk')
    assert isinstance(lm_format_dset.n_tk, int)
    assert lm_format_dset.n_tk > 0


def test_formatting_wnli(max_seq_len: int, stride: int) -> None:
  """Format WNLI datasets correctly."""
  for ver in WNLIDset.vers:
    dset = WNLIDset(ver=ver)

    tknzr = lmp.tknzr.WsTknzr()
    tknzr.build_vocab(batch_txt=dset)

    lm_format_dset = lmp.util.dset.LMFormatDset(
      dset=dset,
      max_seq_len=max_seq_len,
      stride=stride,
      tknzr=tknzr,
    )

    dset_size = len(lm_format_dset)
    assert isinstance(dset_size, int)

    for idx in range(dset_size):
      spl = lm_format_dset[idx]
      assert isinstance(spl, tuple)
      assert len(spl) == 2

      cur_tkids, next_tkids = spl
      assert isinstance(cur_tkids, torch.Tensor)
      assert cur_tkids.dtype == torch.long
      assert isinstance(next_tkids, torch.Tensor)
      assert next_tkids.dtype == torch.long
      assert torch.all(cur_tkids[1:] == next_tkids[:-1])

    assert hasattr(lm_format_dset, 'n_tk')
    assert isinstance(lm_format_dset.n_tk, int)
    assert lm_format_dset.n_tk > 0
