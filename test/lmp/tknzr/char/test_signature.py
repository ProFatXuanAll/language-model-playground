"""Test :py:class:`lmp.tknzr.CharTknzr` signatures."""

import inspect
from typing import get_type_hints

from lmp.tknzr import BaseTknzr, CharTknzr


def test_class() -> None:
  """Ensure class signatures."""
  assert inspect.isclass(CharTknzr)
  assert not inspect.isabstract(CharTknzr)
  assert issubclass(CharTknzr, BaseTknzr)


def test_class_attribute() -> None:
  """Ensure class attributes' signatures."""
  assert get_type_hints(CharTknzr) == get_type_hints(BaseTknzr)
  assert CharTknzr.bos_tk == BaseTknzr.bos_tk
  assert CharTknzr.bos_tkid == BaseTknzr.bos_tkid
  assert CharTknzr.eos_tk == BaseTknzr.eos_tk
  assert CharTknzr.eos_tkid == BaseTknzr.eos_tkid
  assert CharTknzr.file_name == BaseTknzr.file_name
  assert CharTknzr.pad_tk == BaseTknzr.pad_tk
  assert CharTknzr.pad_tkid == BaseTknzr.pad_tkid
  assert CharTknzr.tknzr_name == 'character'
  assert CharTknzr.unk_tk == BaseTknzr.unk_tk
  assert CharTknzr.unk_tkid == BaseTknzr.unk_tkid


def test_inherent_class_method() -> None:
  """Ensure inherent class methods are same as baseclass."""
  assert inspect.signature(CharTknzr.load) == inspect.signature(BaseTknzr.load)
  assert inspect.signature(CharTknzr.train_parser) == inspect.signature(BaseTknzr.train_parser)


def test_inherent_instance_method() -> None:
  """Ensure inherent instance methods are same as baseclass."""
  assert CharTknzr.__init__ == BaseTknzr.__init__
  assert CharTknzr.batch_dec == BaseTknzr.batch_dec
  assert CharTknzr.batch_enc == BaseTknzr.batch_enc
  assert CharTknzr.build_vocab == BaseTknzr.build_vocab
  assert CharTknzr.dec == BaseTknzr.dec
  assert CharTknzr.enc == BaseTknzr.enc
  assert CharTknzr.save == BaseTknzr.save
  assert CharTknzr.norm == BaseTknzr.norm
  assert CharTknzr.vocab_size == BaseTknzr.vocab_size


def test_inherent_static_method() -> None:
  """Ensure inherent static methods are same as baseclass."""
  assert CharTknzr.pad_to_max == BaseTknzr.pad_to_max
  assert CharTknzr.trunc_to_max == BaseTknzr.trunc_to_max


def test_instance_method() -> None:
  """Ensure instance methods' signatures."""
  assert inspect.signature(CharTknzr.tknz) == inspect.signature(BaseTknzr.tknz)
  assert inspect.signature(CharTknzr.dtknz) == inspect.signature(BaseTknzr.dtknz)
