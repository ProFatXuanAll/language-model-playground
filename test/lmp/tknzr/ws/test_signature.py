"""Test :py:class:`lmp.tknzr.WsTknzr` signatures."""

import inspect
from typing import get_type_hints

from lmp.tknzr import BaseTknzr, WsTknzr


def test_class() -> None:
  """Ensure class signatures."""
  assert inspect.isclass(WsTknzr)
  assert not inspect.isabstract(WsTknzr)
  assert issubclass(WsTknzr, BaseTknzr)


def test_class_attribute() -> None:
  """Ensure class attributes' signatures."""
  assert get_type_hints(WsTknzr) == get_type_hints(BaseTknzr)
  assert WsTknzr.bos_tk == BaseTknzr.bos_tk
  assert WsTknzr.bos_tkid == BaseTknzr.bos_tkid
  assert WsTknzr.eos_tk == BaseTknzr.eos_tk
  assert WsTknzr.eos_tkid == BaseTknzr.eos_tkid
  assert WsTknzr.file_name == BaseTknzr.file_name
  assert WsTknzr.pad_tk == BaseTknzr.pad_tk
  assert WsTknzr.pad_tkid == BaseTknzr.pad_tkid
  assert WsTknzr.tknzr_name == 'whitespace'
  assert WsTknzr.unk_tk == BaseTknzr.unk_tk
  assert WsTknzr.unk_tkid == BaseTknzr.unk_tkid


def test_inherent_class_method() -> None:
  """Ensure inherent class methods are same as baseclass."""
  assert inspect.signature(WsTknzr.load) == inspect.signature(BaseTknzr.load)
  assert inspect.signature(WsTknzr.train_parser) == inspect.signature(BaseTknzr.train_parser)


def test_inherent_instance_method() -> None:
  """Ensure inherent instance methods are same as baseclass."""
  assert WsTknzr.__init__ == BaseTknzr.__init__
  assert WsTknzr.batch_dec == BaseTknzr.batch_dec
  assert WsTknzr.batch_enc == BaseTknzr.batch_enc
  assert WsTknzr.build_vocab == BaseTknzr.build_vocab
  assert WsTknzr.dec == BaseTknzr.dec
  assert WsTknzr.enc == BaseTknzr.enc
  assert WsTknzr.save == BaseTknzr.save
  assert WsTknzr.norm == BaseTknzr.norm
  assert WsTknzr.vocab_size == BaseTknzr.vocab_size


def test_inherent_static_method() -> None:
  """Ensure inherent static methods are same as baseclass."""
  assert WsTknzr.pad_to_max == BaseTknzr.pad_to_max
  assert WsTknzr.trunc_to_max == BaseTknzr.trunc_to_max


def test_instance_method() -> None:
  """Ensure instance methods' signatures."""
  assert inspect.signature(WsTknzr.tknz) == inspect.signature(BaseTknzr.tknz)
  assert inspect.signature(WsTknzr.dtknz) == inspect.signature(BaseTknzr.dtknz)
