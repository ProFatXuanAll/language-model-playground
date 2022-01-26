"""Test :py:class:`lmp.model.LSTM2002` signatures."""

import argparse
import inspect
from inspect import Parameter, Signature
from typing import Any, ClassVar, List, Optional, Tuple, get_type_hints

import torch

from lmp.model import LSTM2002, BaseModel
from lmp.tknzr import BaseTknzr


def test_class() -> None:
  """Ensure abstract class signatures."""
  assert inspect.isclass(LSTM2002)
  assert issubclass(LSTM2002, BaseModel)
  assert not inspect.isabstract(LSTM2002)


def test_class_attribute() -> None:
  """Ensure class attributes' signatures."""
  type_hints = get_type_hints(LSTM2002)
  assert type_hints['model_name'] == ClassVar[str]
  assert LSTM2002.model_name == 'LSTM-2002'


def test_class_method() -> None:
  """Ensure class methods' signatures."""
  assert hasattr(LSTM2002, 'train_parser')
  assert inspect.ismethod(LSTM2002.train_parser)
  assert LSTM2002.train_parser.__self__ == LSTM2002
  assert inspect.signature(LSTM2002.train_parser) == Signature(
    parameters=[
      Parameter(
        name='parser',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=argparse.ArgumentParser,
      ),
    ],
    return_annotation=None,
  )


def test_instance_method() -> None:
  """Ensure instance methods' signatures."""
  assert hasattr(LSTM2002, '__init__')
  assert inspect.isfunction(LSTM2002.__init__)
  assert inspect.signature(LSTM2002.__init__) == Signature(
    parameters=[
      Parameter(
        name='self',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
      ),
      Parameter(
        name='d_cell',
        kind=Parameter.KEYWORD_ONLY,
        default=Parameter.empty,
        annotation=int,
      ),
      Parameter(
        name='d_emb',
        kind=Parameter.KEYWORD_ONLY,
        default=Parameter.empty,
        annotation=int,
      ),
      Parameter(
        name='n_cell',
        kind=Parameter.KEYWORD_ONLY,
        default=Parameter.empty,
        annotation=int,
      ),
      Parameter(
        name='tknzr',
        kind=Parameter.KEYWORD_ONLY,
        default=Parameter.empty,
        annotation=BaseTknzr,
      ),
      Parameter(
        name='kwargs',
        kind=Parameter.VAR_KEYWORD,
        annotation=Any,
      ),
    ],
    return_annotation=Signature.empty,
  )
  assert hasattr(LSTM2002, 'forward')
  assert inspect.isfunction(LSTM2002.forward)
  assert inspect.signature(LSTM2002.forward) == Signature(
    parameters=[
      Parameter(
        name='self',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
      ),
      Parameter(
        name='batch_cur_tkids',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        annotation=torch.Tensor,
      ),
      Parameter(
        name='batch_next_tkids',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        annotation=torch.Tensor,
      ),
    ],
    return_annotation=torch.Tensor,
  )
  assert hasattr(LSTM2002, 'params_init')
  assert inspect.isfunction(LSTM2002.params_init)
  assert inspect.signature(LSTM2002.params_init) == Signature(
    parameters=[
      Parameter(
        name='self',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
      ),
    ],
    return_annotation=None,
  )
  assert hasattr(LSTM2002, 'pred')
  assert inspect.isfunction(LSTM2002.pred)
  assert inspect.signature(LSTM2002.pred) == Signature(
    parameters=[
      Parameter(
        name='self',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
      ),
      Parameter(
        name='batch_cur_tkids',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        annotation=torch.Tensor,
      ),
      Parameter(
        name='batch_prev_states',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        annotation=Optional[List[torch.Tensor]],
        default=None,
      ),
    ],
    return_annotation=Tuple[torch.Tensor, List[torch.Tensor]],
  )
