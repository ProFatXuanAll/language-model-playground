"""Tokenizer utilities."""

from typing import Any

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
  lmp.tknzr
    All available tokenizers.

  Examples
  --------
  >>> from lmp.tknzr import WsTknzr
  >>> import lmp.util.tknzr
  >>> isinstance(
  ...   lmp.util.tknzr.create(
  ...     is_uncased=False,
  ...     max_vocab=-1,
  ...     min_count=0,
  ...     tknzr_name=WsTknzr.tknzr_name,
  ...   ),
  ...   WsTknzr,
  ... )
  True
  """
  # `tknzr_name` validation.
  lmp.util.validate.raise_if_not_instance(val=tknzr_name, val_name='tknzr_name', val_type=str)
  lmp.util.validate.raise_if_not_in(val=tknzr_name, val_name='tknzr_name', val_range=list(TKNZR_OPTS.keys()))

  # `kwargs` validation will be performed in `BaseTknzr.__init__`.
  # Currently `mypy` cannot perform static type check on `**kwargs`, and I think it can only be check by runtime and
  # therefore `mypy` may no be able to solve this issue forever.  So we use `# type: ignore` to silence error.
  return TKNZR_OPTS[tknzr_name](**kwargs)  # type: ignore


def load(exp_name: str, tknzr_name: str) -> BaseTknzr:
  """Load pre-trained tokenizer instance by experiment name.

  Load pre-trained tokenizer from path ``root/exp/exp_name``.  Here ``root`` refers to
  :py:attr:`lmp.util.path.PROJECT_ROOT`.  The type of tokenizer instance is based on ``tknzr_name``.

  Parameters
  ----------
  exp_name: str
    Pre-trained tokenizer experiment name.
  tknzr_name: str
    Name of the tokenizer to be loaded.

  Returns
  -------
  lmp.tknzr.BaseTknzr
    Pre-trained tokenizer instance.

  See Also
  --------
  lmp.tknzr
    All available tokenizers.

  Examples
  --------
  >>> from lmp.tknzr import WsTknzr
  >>> import lmp.util.tknzr
  >>> tknzr = lmp.util.tknzr.create(
  ...   is_uncased=True,
  ...   max_vocab=10,
  ...   min_count=2,
  ...   tknzr_name=WsTknzr.tknzr_name,
  ... )
  >>> tknzr.save(exp_name='my_exp')
  >>> load_tknzr = lmp.util.tknzr.load(exp_name='my_exp', tknzr_name=WsTknzr.tknzr_name)
  >>> isinstance(load_tknzr, WsTknzr)
  True
  >>> load_tknzr.is_uncased == load_tknzr.is_uncased
  True
  >>> load_tknzr.max_vocab == load_tknzr.max_vocab
  True
  >>> load_tknzr.min_count == load_tknzr.min_count
  True
  """
  # `tknzr_name` validation.
  lmp.util.validate.raise_if_not_instance(val=tknzr_name, val_name='tknzr_name', val_type=str)
  lmp.util.validate.raise_if_not_in(val=tknzr_name, val_name='tknzr_name', val_range=list(TKNZR_OPTS.keys()))

  # `exp_name` will be validated in `BaseTknzr.load`.
  return TKNZR_OPTS[tknzr_name].load(exp_name=exp_name)
