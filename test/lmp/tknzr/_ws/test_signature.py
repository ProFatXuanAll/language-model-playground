"""Test :py:mod:`lmp.tknzr._ws` signatures."""

import inspect
import re
from typing import ClassVar, get_type_hints

import lmp.tknzr._ws
from lmp.tknzr._base import BaseTknzr


def test_module_attribute() -> None:
  """Ensure module attributes' signatures."""
  assert hasattr(lmp.tknzr._ws, 'SPLIT_PTTN')
  assert isinstance(lmp.tknzr._ws.SPLIT_PTTN, re.Pattern)
  assert lmp.tknzr._ws.SPLIT_PTTN.pattern == r'(<bos>|<eos>|<pad>|<unk>|\s+)'

  assert hasattr(lmp.tknzr._ws, 'WsTknzr')
  assert inspect.isclass(lmp.tknzr._ws.WsTknzr)
  assert not inspect.isabstract(lmp.tknzr._ws.WsTknzr)
  assert issubclass(lmp.tknzr._ws.WsTknzr, BaseTknzr)


def test_class_attribute() -> None:
  """Ensure class attributes' signatures."""
  assert get_type_hints(lmp.tknzr._ws.WsTknzr) == {'tknzr_name': ClassVar[str]}
  assert lmp.tknzr._ws.WsTknzr.tknzr_name == 'whitespace'


def test_inherent_class_method() -> None:
  """Ensure inherent class methods are same as base class."""
  assert inspect.signature(lmp.tknzr._ws.WsTknzr.add_CLI_args) == inspect.signature(BaseTknzr.add_CLI_args)


def test_inherent_instance_method() -> None:
  """Ensure inherent instance methods are same as base class."""
  assert lmp.tknzr._ws.WsTknzr.__init__ == BaseTknzr.__init__
  assert lmp.tknzr._ws.WsTknzr.build_vocab == BaseTknzr.build_vocab
  assert lmp.tknzr._ws.WsTknzr.dec == BaseTknzr.dec
  assert lmp.tknzr._ws.WsTknzr.enc == BaseTknzr.enc
  assert lmp.tknzr._ws.WsTknzr.pad_to_max == BaseTknzr.pad_to_max
  assert lmp.tknzr._ws.WsTknzr.norm == BaseTknzr.norm
  assert lmp.tknzr._ws.WsTknzr.vocab_size == BaseTknzr.vocab_size


def test_instance_method() -> None:
  """Ensure instance methods' signatures."""
  assert inspect.signature(lmp.tknzr._ws.WsTknzr.dtknz) == inspect.signature(BaseTknzr.dtknz)
  assert inspect.signature(lmp.tknzr._ws.WsTknzr.tknz) == inspect.signature(BaseTknzr.tknz)
