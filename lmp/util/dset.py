"""Dataset utilities."""

from typing import Any, Iterator

import torch.utils.data

import lmp.util.validate
from lmp.dset import DSET_OPTS, BaseDset
from lmp.tknzr import BaseTknzr


class FastTensorDset(torch.utils.data.Dataset):
  """Fast version of tensor dataset.

  We use :py:meth:`lmp.tknzr.BaseTknzr.batch_enc` to convert text samples of :py:class:`lmp.dset.BaseDset` into token
  id tensors.  We save conversion results instead of converting on the fly.  Thus dataset passed in will be completely
  loaded into memory.

  .. danger::

     This datset will consume huge memory.  Make sure you do this only if you have enough memory.

  Parameters
  ----------
  dset: lmp.dset.BaseDset
    Dataset instance to be converted.
  max_seq_len: int
    Maximum length constraint.
  tknzr: lmp.tknzr.BaseTknzr
    Tokenizer instance to convert text into token id list.

  Attributes
  ----------
  spl: torch.Tensor
    Tensor represent all token id lists in the dataset.  Note that we do not use tensor list, for which will trigger
    complete memory copy on every worker process since Python's refcount make Python process copy on access.  See the
    following link for details:  https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
  """

  def __init__(self, *, dset: BaseDset, max_seq_len: int, tknzr: BaseTknzr):
    # Since CUDA only support integer with Long type, we use `torch.LongTensor` instead of `torch.IntTensor`.
    self.spls = torch.LongTensor(tknzr.batch_enc(batch_txt=dset.spls, max_seq_len=max_seq_len))

  def __getitem__(self, idx: int) -> torch.Tensor:
    """Sample tensor of token id list using index.

    Parameters
    ----------
    idx: int
      Sample index.

    Returns
    -------
    torch.Tensor
      The sample which index equals to ``idx``.
    """
    return self.spls[idx]

  def __iter__(self) -> Iterator[torch.Tensor]:
    """Iterate through each sample in the dataset.

    Yields
    ------
    torch.Tensor
      One sample in ``self.spls``, ordered by sample indices.
    """
    for i in range(self.spls.size(0)):
      yield self.spls[i]

  def __len__(self) -> int:
    """Get dataset size.

    Returns
    -------
    int
      Number of samples in the dataset.
    """
    return self.spls.size(0)


class SlowTensorDset(torch.utils.data.Dataset):
  """Slow version of tensor dataset.

  We use :py:meth:`lmp.tknzr.BaseTknzr.enc` to convert text samples of :py:class:`lmp.dset.BaseDset` into token id
  tensors.  We convert text samples on the fly and do not save conversion results.  The sampling speed of this dataset
  is 10x slower compare to :py:class:`lmp.util.dset.FastTensorDset`.  But this is the only class offering sampling
  mechanism for huge dataset (which cannot be loaded in memory completely).

  Parameters
  ----------
  dset: lmp.dset.BaseDset
    Dataset instance to be converted.
  max_seq_len: int
    Maximum length constraint.
  tknzr: lmp.tknzr.BaseTknzr
    Tokenizer instance to convert text into token id list.

  Attributes
  ----------
  dset: lmp.dset.BaseDset
    Dataset instance to be converted.
  max_seq_len: int
    Maximum length constraint.
  tknzr: lmp.tknzr.BaseTknzr
    Tokenizer instance to convert text into token id list.
  """

  def __init__(self, *, dset: BaseDset, max_seq_len: int, tknzr: BaseTknzr):
    self.dset = dset
    self.max_seq_len = max_seq_len
    self.tknzr = tknzr

  def __getitem__(self, idx: int) -> torch.Tensor:
    """Sample tensor of token id list using index.

    Parameters
    ----------
    idx: int
      Sample index.

    Returns
    -------
    torch.Tensor
      The sample which index equals to ``idx``.
    """
    # Since CUDA only support integer with Long type, we use `torch.LongTensor` instead of `torch.IntTensor`.
    return torch.LongTensor(self.tknzr.enc(max_seq_len=self.max_seq_len, txt=self.dset[idx]))

  def __iter__(self) -> Iterator[torch.Tensor]:
    """Iterate through each sample in the dataset.

    Yields
    ------
    torch.Tensor
      One tensor of converted token id list, ordered by the original ordered in ``self.dset``.
    """
    for i in range(len(self)):
      yield self[i]

  def __len__(self) -> int:
    """Get dataset size.

    Returns
    -------
    int
      Number of samples in the dataset.
    """
    return len(self.dset)


def load(dset_name: str, ver: str, **kwargs: Any) -> BaseDset:
  """Load dataset.

  Parameters
  ----------
  dset_name: str
    Name of the dataset to be loaded.
  ver: str
    Version of the dataset to be loaded.

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
