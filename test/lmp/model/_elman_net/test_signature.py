"""Test :py:mod:`lmp.model._elman_net` signatures."""

import inspect
from inspect import Parameter, Signature
from typing import Any, ClassVar, get_type_hints

import lmp.model._elman_net
from lmp.model._base import BaseModel
from lmp.tknzr._base import BaseTknzr


def test_module_attribute() -> None:
  """Ensure module attributes' signatures."""
  assert hasattr(lmp.model._elman_net, 'ElmanNet')
  assert inspect.isclass(lmp.model._elman_net.ElmanNet)
  assert issubclass(lmp.model._elman_net.ElmanNet, BaseModel)
  assert not inspect.isabstract(lmp.model._elman_net.ElmanNet)


def test_class_attribute() -> None:
  """Ensure class attributes' signatures."""
  type_hints = get_type_hints(lmp.model._elman_net.ElmanNet)
  assert type_hints['model_name'] == ClassVar[str]
  assert lmp.model._elman_net.ElmanNet.model_name == 'Elman-Net'


def test_class_method() -> None:
  """Ensure class methods' signatures."""
  assert hasattr(lmp.model._elman_net.ElmanNet, 'add_CLI_args')
  assert inspect.ismethod(lmp.model._elman_net.ElmanNet.add_CLI_args)
  assert lmp.model._elman_net.ElmanNet.add_CLI_args.__self__ == lmp.model._elman_net.ElmanNet
  assert inspect.signature(lmp.model._elman_net.ElmanNet.add_CLI_args) == inspect.signature(BaseModel.add_CLI_args)


def test_instance_method() -> None:
  """Ensure instance methods' signatures."""
  assert hasattr(lmp.model._elman_net.ElmanNet, '__init__')
  assert inspect.isfunction(lmp.model._elman_net.ElmanNet.__init__)
  assert inspect.signature(lmp.model._elman_net.ElmanNet.__init__) == Signature(
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
        name='d_hid',
        kind=Parameter.KEYWORD_ONLY,
        default=Parameter.empty,
        annotation=int,
      ),
      Parameter(
        name='p_emb',
        kind=Parameter.KEYWORD_ONLY,
        default=Parameter.empty,
        annotation=float,
      ),
      Parameter(
        name='p_hid',
        kind=Parameter.KEYWORD_ONLY,
        default=Parameter.empty,
        annotation=float,
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

  assert hasattr(lmp.model._elman_net.ElmanNet, 'forward')
  assert inspect.isfunction(lmp.model._elman_net.ElmanNet.forward)
  assert inspect.signature(lmp.model._elman_net.ElmanNet.forward) == inspect.signature(BaseModel.forward)

  assert hasattr(lmp.model._elman_net.ElmanNet, 'loss')
  assert inspect.isfunction(lmp.model._elman_net.ElmanNet.loss)
  assert inspect.signature(lmp.model._elman_net.ElmanNet.loss) == inspect.signature(BaseModel.loss)

  assert hasattr(lmp.model._elman_net.ElmanNet, 'params_init')
  assert inspect.isfunction(lmp.model._elman_net.ElmanNet.params_init)
  assert inspect.signature(lmp.model._elman_net.ElmanNet.params_init) == inspect.signature(BaseModel.params_init)

  assert hasattr(lmp.model._elman_net.ElmanNet, 'pred')
  assert inspect.isfunction(lmp.model._elman_net.ElmanNet.pred)
  assert inspect.signature(lmp.model._elman_net.ElmanNet.pred) == inspect.signature(BaseModel.pred)
