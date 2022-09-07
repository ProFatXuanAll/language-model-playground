"""Dataset utilities."""

from typing import Any, Tuple

import torch
import torch.utils.data

import lmp.util.validate
from lmp.dset import DSET_OPTS, BaseDset
from lmp.tknzr import BaseTknzr
from lmp.vars import EOS_TKID


class LMFormatDset(torch.utils.data.Dataset):
  """Convert dataset into language model training format.

  Each dataset samples is converted into token id sequence.
  Token id sequence is splitted into multiple subsequences.
  All subsequences have the same length.

  Parameters
  ----------
  max_seq_len: int
    Context window size applied on dataset samples.
  stride: int
    Context windows may have overlaps.
    Number of overlapping tokens between subsamples is called stride.

  Attributes
  ----------
  batch_cur_tkids: list[torch.Tensor]
    Language model input token ids.
  batch_is_not_ctx: list[torch.Tensor]
    Boolean tensor indicate whether token ids are used as conditional context or not.
    Conditional context means tokens that are overlapping with other context windows.
  batch_next_tkids: list[torch.Tensor]
    Language model prediction target.
  n_tk: int
    Number of tokens in the dataset.
    Overlapping tokens are not repeatly counted.
    Padding tokens are not counted.
  """

  def __init__(self, *, dset: BaseDset, max_seq_len: int, stride: int, tknzr: BaseTknzr):
    # `dset` validation.
    lmp.util.validate.raise_if_not_instance(val=dset, val_name='dset', val_type=BaseDset)
    # `max_seq_len` and `stride` validation.
    lmp.util.validate.raise_if_not_instance(val=max_seq_len, val_name='max_seq_len', val_type=int)
    lmp.util.validate.raise_if_not_instance(val=stride, val_name='stride', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, stride, max_seq_len], val_names=['1', 'stride', 'max_seq_len'])
    # `tknzr` validation.
    lmp.util.validate.raise_if_not_instance(val=tknzr, val_name='tknzr', val_type=BaseTknzr)

    self.batch_is_not_ctx = []
    self.batch_cur_tkids = []
    self.batch_next_tkids = []
    self.n_tk = 0

    # `ctx_len` is the length of conditional context token ids.
    # `tgt_len` is the length of target token ids.
    ctx_len = max(0, max_seq_len - stride)
    tgt_len = max_seq_len - ctx_len

    # Slice token sequence into context windows with each length equals to `max_seq_len`.
    # Context windows shorter than `max_seq_len` are padded at the end.
    # Context windows may have overlaps.
    # Number of overlapping tokens are determined by `stride`.
    for txt in dset:
      # Encode text and record number of tokens.
      tkids = tknzr.enc(txt=txt)
      self.n_tk += len(tkids)

      for idx in range(0, len(tkids), stride):
        cur_tkids = tknzr.pad_to_max(max_seq_len=max_seq_len, tkids=tkids[idx:idx + max_seq_len])
        next_tkids = tknzr.pad_to_max(max_seq_len=max_seq_len, tkids=tkids[idx + 1:idx + 1 + max_seq_len])

        # Skip sequence starting with EOS.
        if cur_tkids[0] == EOS_TKID:
          continue

        # If we share tensor's memory here, we can save huge memory consumption.
        # However this means every time we need to fetch different parts of shared memory we have to index.
        # And Pytorch is such at indexing (much slower on CUDA then CPU).
        # Thus to make computation fast, we do not share tensor's memory here.
        self.batch_cur_tkids.append(torch.LongTensor(cur_tkids))
        self.batch_next_tkids.append(torch.LongTensor(next_tkids))

        # All token are not used as conditional context at the start since it is just started.
        if idx == 0:
          self.batch_is_not_ctx.append(torch.tensor([True] * max_seq_len))
        else:
          self.batch_is_not_ctx.append(torch.tensor([False] * ctx_len + [True] * tgt_len))

  def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample context window using index.

    Parameters
    ----------
    idx: int
      Sample index.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
      The context window whose index equals to ``idx``.
      The first tensor in the returned tuple represent whether a token is used as conditional context.
      The second tensor in the returned tuple represent the current context window.
      The third tensor in the returned tuple represent the prediction target of the current context window.
    """
    return self.batch_is_not_ctx[idx], self.batch_cur_tkids[idx], self.batch_next_tkids[idx]

  def __len__(self) -> int:
    """Get dataset size.

    Returns
    -------
    int
      Number of context window in the dataset.
    """
    return len(self.batch_cur_tkids)


def load(dset_name: str, ver: str, **kwargs: Any) -> BaseDset:
  """Load dataset.

  Parameters
  ----------
  dset_name: str
    Name of the dataset to load.
  ver: str
    Version of the dataset to load.

  Returns
  -------
  lmp.dset.BaseDset
    Loaded dataset instance.

  See Also
  --------
  :doc:`lmp.dset </dset/index>`
    All available datasets.

  Examples
  --------
  >>> from lmp.dset import WikiText2Dset
  >>> import lmp.util.dset
  >>> dset = lmp.util.dset.load(dset_name='wiki-text-2', ver='train')
  >>> isinstance(dset, WikiText2Dset)
  True
  >>> dset.ver == 'train'
  True
  """
  # `dset_name` validation.
  lmp.util.validate.raise_if_not_instance(val=dset_name, val_name='dset_name', val_type=str)
  lmp.util.validate.raise_if_not_in(val=dset_name, val_name='dset_name', val_range=list(DSET_OPTS.keys()))

  # `ver` will be validated in `BaseDset.__init__`.
  return DSET_OPTS[dset_name](ver=ver)
