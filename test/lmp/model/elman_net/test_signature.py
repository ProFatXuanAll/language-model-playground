"""Test :py:class:`lmp.model.ElmanNet` signatures."""

import argparse
import inspect
from inspect import Parameter, Signature
from typing import Any, ClassVar, Optional, Tuple, get_type_hints

import torch

from lmp.model import BaseModel, ElmanNet
from lmp.tknzr import BaseTknzr


def test_class() -> None:
  """Ensure abstract class signatures."""
  assert inspect.isclass(ElmanNet)
  assert issubclass(ElmanNet, BaseModel)
  assert not inspect.isabstract(ElmanNet)


def test_class_attribute() -> None:
  """Ensure class attributes' signatures."""
  type_hints = get_type_hints(ElmanNet)
  assert type_hints['model_name'] == ClassVar[str]
  assert ElmanNet.model_name == 'Elman-Net'


def test_class_method() -> None:
  """Ensure class methods' signatures."""
  assert hasattr(ElmanNet, 'train_parser')
  assert inspect.ismethod(ElmanNet.train_parser)
  assert ElmanNet.train_parser.__self__ == ElmanNet
  assert inspect.signature(ElmanNet.train_parser) == Signature(
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
  assert hasattr(ElmanNet, '__init__')
  assert inspect.isfunction(ElmanNet.__init__)
  assert inspect.signature(ElmanNet.__init__) == Signature(
    parameters=[
      Parameter(
        name='self',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
      ),
      Parameter(
        name='d_emb',
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
  assert hasattr(ElmanNet, 'forward')
  assert inspect.isfunction(ElmanNet.forward)
  assert inspect.signature(ElmanNet.forward) == Signature(
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
  assert hasattr(ElmanNet, 'pred')
  assert inspect.isfunction(ElmanNet.pred)
  assert inspect.signature(ElmanNet.pred) == Signature(
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
        annotation=Optional[torch.Tensor],
        default=None,
      ),
    ],
    return_annotation=Tuple[torch.Tensor, torch.Tensor],
  )
