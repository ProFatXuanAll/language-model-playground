"""Test :py:mod:`lmp.tknzr._char` signatures."""

import inspect
import re
from typing import ClassVar, get_type_hints

import lmp.tknzr._char
from lmp.tknzr._base import BaseTknzr


def test_module_attribute() -> None:
  """Ensure module attributes' signatures."""
  assert hasattr(lmp.tknzr._char, 'SP_TKS_PTTN')
  assert isinstance(lmp.tknzr._char.SP_TKS_PTTN, re.Pattern)
  assert lmp.tknzr._char.SP_TKS_PTTN.pattern == r'(<bos>|<eos>|<pad>|<unk>)'

  assert hasattr(lmp.tknzr._char, 'CharTknzr')
  assert inspect.isclass(lmp.tknzr._char.CharTknzr)
  assert not inspect.isabstract(lmp.tknzr._char.CharTknzr)
  assert issubclass(lmp.tknzr._char.CharTknzr, BaseTknzr)


def test_class_attribute() -> None:
  """Ensure class attributes' signatures."""
  assert get_type_hints(lmp.tknzr._char.CharTknzr) == {'tknzr_name': ClassVar[str]}
  assert lmp.tknzr._char.CharTknzr.tknzr_name == 'character'


def test_inherent_class_method() -> None:
  """Ensure inherent class methods are same as base class."""
  assert inspect.signature(lmp.tknzr._char.CharTknzr.add_CLI_args) == inspect.signature(BaseTknzr.add_CLI_args)


def test_inherent_instance_method() -> None:
  """Ensure inherent instance methods are same as base class."""
  assert lmp.tknzr._char.CharTknzr.__init__ == BaseTknzr.__init__
  assert lmp.tknzr._char.CharTknzr.build_vocab == BaseTknzr.build_vocab
  assert lmp.tknzr._char.CharTknzr.dec == BaseTknzr.dec
  assert lmp.tknzr._char.CharTknzr.enc == BaseTknzr.enc
  assert lmp.tknzr._char.CharTknzr.pad_to_max == BaseTknzr.pad_to_max
  assert lmp.tknzr._char.CharTknzr.norm == BaseTknzr.norm
  assert lmp.tknzr._char.CharTknzr.vocab_size == BaseTknzr.vocab_size


def test_instance_method() -> None:
  """Ensure instance methods' signatures."""
  assert inspect.signature(lmp.tknzr._char.CharTknzr.dtknz) == inspect.signature(BaseTknzr.dtknz)
  assert inspect.signature(lmp.tknzr._char.CharTknzr.tknz) == inspect.signature(BaseTknzr.tknz)
