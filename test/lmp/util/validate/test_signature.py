"""Test :py:mod:`lmp.util.validate` signatures."""

import inspect
from inspect import Parameter, Signature
from typing import Any, List, Type, Union

import lmp.util.validate


def test_module_method():
  """Ensure module methods' signatures."""
  assert hasattr(lmp.util.validate, 'raise_if_not_instance')
  assert inspect.isfunction(lmp.util.validate.raise_if_not_instance)
  assert inspect.signature(lmp.util.validate.raise_if_not_instance) == Signature(
    parameters=[
      Parameter(
        name='val',
        kind=Parameter.KEYWORD_ONLY,
        default=Parameter.empty,
        annotation=Any,
      ),
      Parameter(
        name='val_name',
        kind=Parameter.KEYWORD_ONLY,
        default=Parameter.empty,
        annotation=str,
      ),
      Parameter(
        name='val_type',
        kind=Parameter.KEYWORD_ONLY,
        default=Parameter.empty,
        annotation=Type,
      ),
    ],
    return_annotation=None,
  )
  assert hasattr(lmp.util.validate, 'raise_if_empty_str')
  assert inspect.isfunction(lmp.util.validate.raise_if_empty_str)
  assert inspect.signature(lmp.util.validate.raise_if_empty_str) == Signature(
    parameters=[
      Parameter(
        name='val',
        kind=Parameter.KEYWORD_ONLY,
        default=Parameter.empty,
        annotation=str,
      ),
      Parameter(
        name='val_name',
        kind=Parameter.KEYWORD_ONLY,
        default=Parameter.empty,
        annotation=str,
      ),
    ],
    return_annotation=None,
  )
  assert hasattr(lmp.util.validate, 'raise_if_not_in')
  assert inspect.isfunction(lmp.util.validate.raise_if_not_in)
  assert inspect.signature(lmp.util.validate.raise_if_not_in) == Signature(
    parameters=[
      Parameter(
        name='val',
        kind=Parameter.KEYWORD_ONLY,
        default=Parameter.empty,
        annotation=Any,
      ),
      Parameter(
        name='val_name',
        kind=Parameter.KEYWORD_ONLY,
        default=Parameter.empty,
        annotation=str,
      ),
      Parameter(
        name='val_range',
        kind=Parameter.KEYWORD_ONLY,
        default=Parameter.empty,
        annotation=List,
      ),
    ],
    return_annotation=None,
  )
  assert hasattr(lmp.util.validate, 'raise_if_is_directory')
  assert inspect.isfunction(lmp.util.validate.raise_if_is_directory)
  assert inspect.signature(lmp.util.validate.raise_if_is_directory) == Signature(
    parameters=[
      Parameter(
        name='path',
        kind=Parameter.KEYWORD_ONLY,
        default=Parameter.empty,
        annotation=str,
      ),
    ],
    return_annotation=None,
  )
  assert hasattr(lmp.util.validate, 'raise_if_is_file')
  assert inspect.isfunction(lmp.util.validate.raise_if_is_file)
  assert inspect.signature(lmp.util.validate.raise_if_is_file) == Signature(
    parameters=[
      Parameter(
        name='path',
        kind=Parameter.KEYWORD_ONLY,
        default=Parameter.empty,
        annotation=str,
      ),
    ],
    return_annotation=None,
  )
  assert hasattr(lmp.util.validate, 'raise_if_wrong_ordered')
  assert inspect.isfunction(lmp.util.validate.raise_if_wrong_ordered)
  assert inspect.signature(lmp.util.validate.raise_if_wrong_ordered) == Signature(
    parameters=[
      Parameter(
        name='vals',
        kind=Parameter.KEYWORD_ONLY,
        default=Parameter.empty,
        annotation=List[Union[float, int]],
      ),
      Parameter(
        name='val_names',
        kind=Parameter.KEYWORD_ONLY,
        default=Parameter.empty,
        annotation=List[str],
      ),
    ],
    return_annotation=None,
  )
