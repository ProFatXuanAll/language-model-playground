"""Tokenizer utilities."""

import os
import pickle
from typing import Any

import lmp.util.path
import lmp.util.validate
from lmp.tknzr import TKNZR_OPTS, BaseTknzr


def create(tknzr_name: str, **kwargs: Any) -> BaseTknzr:
  """Create tokenizer instance by tokenizer's name.

  Tokenizer's arguments are collected in ``**kwargs`` and are passed directly to tokenizer's constructor.

  Parameters
  ----------
  tknzr_name: str
    Name of the tokenizer to create.
  kwargs: typing.Any, optional
    Tokenizer's parameters.

  Returns
  -------
  lmp.tknzr.BaseTknzr
    Tokenizer instance.

  See Also
  --------
  :doc:`lmp.tknzr </tknzr/index>`
    All available tokenizers.

  Examples
  --------
  >>> from lmp.tknzr import WsTknzr
  >>> import lmp.util.tknzr
  >>> tknzr = lmp.util.tknzr.create(
  ...   is_uncased=False,
  ...   max_seq_len=128,
  ...   max_vocab=-1,
  ...   min_count=0,
  ...   tknzr_name=WsTknzr.tknzr_name,
  ... )
  >>> assert isinstance(tknzr, WsTknzr)
  """
  # `tknzr_name` validation.
  lmp.util.validate.raise_if_not_instance(val=tknzr_name, val_name='tknzr_name', val_type=str)
  lmp.util.validate.raise_if_not_in(val=tknzr_name, val_name='tknzr_name', val_range=list(TKNZR_OPTS.keys()))

  # `kwargs` validation will be performed in `BaseTknzr.__init__`.
  # Currently `mypy` cannot perform static type check on `**kwargs`, and I think it can only be check by runtime and
  # therefore `mypy` may no be able to solve this issue forever.  So we use `# type: ignore` to silence error.
  return TKNZR_OPTS[tknzr_name](**kwargs)  # type: ignore


def load(exp_name: str) -> BaseTknzr:
  """Load pre-trained tokenizer instance by experiment name.

  Load pre-trained tokenizer from path ``project_root/exp/exp_name``.

  Parameters
  ----------
  exp_name: str
    Pre-trained tokenizer experiment name.

  Returns
  -------
  lmp.tknzr.BaseTknzr
    Pre-trained tokenizer instance.

  See Also
  --------
  :doc:`lmp.tknzr </tknzr/index>`
    All available tokenizers.

  Examples
  --------
  >>> from lmp.tknzr import WsTknzr
  >>> import lmp.util.tknzr
  >>> tknzr = lmp.util.tknzr.create(
  ...   is_uncased=True,
  ...   max_seq_len=128,
  ...   max_vocab=10,
  ...   min_count=2,
  ...   tknzr_name=WsTknzr.tknzr_name,
  ... )
  >>> tknzr.save(exp_name='my_exp')
  >>> load_tknzr = lmp.util.tknzr.load(exp_name='my_exp')
  >>> assert isinstance(load_tknzr, WsTknzr)
  >>> assert load_tknzr.id2tk == tknzr.id2tk
  >>> assert load_tknzr.is_uncased == tknzr.is_uncased
  >>> assert load_tknzr.max_seq_len == tknzr.max_seq_len
  >>> assert load_tknzr.max_vocab == tknzr.max_vocab
  >>> assert load_tknzr.min_count == tknzr.min_count
  >>> assert load_tknzr.tk2id == tknzr.tk2id
  """
  # `exp_name` validation.
  lmp.util.validate.raise_if_not_instance(val=exp_name, val_name='exp_name', val_type=str)
  lmp.util.validate.raise_if_empty_str(val=exp_name, val_name='exp_name')

  # `file_path` validation
  file_path = os.path.join(lmp.util.path.EXP_PATH, exp_name, 'tknzr.pkl')
  lmp.util.validate.raise_if_is_directory(path=file_path)

  # Load tokenizer from pickle.
  with open(file_path, 'rb') as f:
    tknzr = pickle.load(f)
  return tknzr


def save(exp_name: str, tknzr: BaseTknzr) -> None:
  """Save tokenizer as pickle file.

  .. danger::

     This method overwrite existed files.  Make sure you know what you are doing before calling this method.

  Parameters
  ----------
  exp_name: int
    Tokenizer training experiment name.
  tknzr: lmp.model.BaseTknzr
    Tokenizer to be saved.

  Returns
  -------
  None

  See Also
  --------
  lmp.util.tknzr.load
    Load pre-trained tokenizer instance by experiment name.

  Examples
  --------
  >>> from lmp.tknzr import CharTknzr
  >>> import lmp.util.tknzr
  >>> tknzr = CharTknzr(is_uncased=False, max_vocab=10, min_count=2)
  >>> lmp.util.tknzr.save(exp_name='test', tknzr=tknzr)
  None
  """
  # `exp_name` validation.
  lmp.util.validate.raise_if_not_instance(val=exp_name, val_name='exp_name', val_type=str)
  lmp.util.validate.raise_if_empty_str(val=exp_name, val_name='exp_name')

  # `dir_path` validation
  dir_path = os.path.join(lmp.util.path.EXP_PATH, exp_name)
  lmp.util.validate.raise_if_is_file(path=dir_path)

  if not os.path.exists(dir_path):
    os.makedirs(dir_path)

  # `file_path` validation.
  file_path = os.path.join(dir_path, 'tknzr.pkl')
  lmp.util.validate.raise_if_is_directory(path=file_path)

  # Save tokenizer as pickle.
  with open(file_path, 'wb') as f:
    pickle.dump(tknzr, f)
