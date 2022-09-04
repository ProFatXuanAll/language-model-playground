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
        annotation=Any,
        default=Parameter.empty,
        kind=Parameter.KEYWORD_ONLY,
        name='val',
      ),
      Parameter(
        annotation=str,
        default=Parameter.empty,
        kind=Parameter.KEYWORD_ONLY,
        name='val_name',
      ),
      Parameter(
        annotation=Type,
        default=Parameter.empty,
        kind=Parameter.KEYWORD_ONLY,
        name='val_type',
      ),
    ],
    return_annotation=None,
  )

  assert hasattr(lmp.util.validate, 'raise_if_empty_str')
  assert inspect.isfunction(lmp.util.validate.raise_if_empty_str)
  assert inspect.signature(lmp.util.validate.raise_if_empty_str) == Signature(
    parameters=[
      Parameter(
        annotation=str,
        default=Parameter.empty,
        kind=Parameter.KEYWORD_ONLY,
        name='val',
      ),
      Parameter(
        annotation=str,
        default=Parameter.empty,
        kind=Parameter.KEYWORD_ONLY,
        name='val_name',
      ),
    ],
    return_annotation=None,
  )

  assert hasattr(lmp.util.validate, 'raise_if_not_in')
  assert inspect.isfunction(lmp.util.validate.raise_if_not_in)
  assert inspect.signature(lmp.util.validate.raise_if_not_in) == Signature(
    parameters=[
      Parameter(
        annotation=Any,
        default=Parameter.empty,
        kind=Parameter.KEYWORD_ONLY,
        name='val',
      ),
      Parameter(
        annotation=str,
        default=Parameter.empty,
        kind=Parameter.KEYWORD_ONLY,
        name='val_name',
      ),
      Parameter(
        annotation=List,
        default=Parameter.empty,
        kind=Parameter.KEYWORD_ONLY,
        name='val_range',
      ),
    ],
    return_annotation=None,
  )

  assert hasattr(lmp.util.validate, 'raise_if_is_directory')
  assert inspect.isfunction(lmp.util.validate.raise_if_is_directory)
  assert inspect.signature(lmp.util.validate.raise_if_is_directory) == Signature(
    parameters=[
      Parameter(
        annotation=str,
        default=Parameter.empty,
        kind=Parameter.KEYWORD_ONLY,
        name='path',
      ),
    ],
    return_annotation=None,
  )

  assert hasattr(lmp.util.validate, 'raise_if_is_file')
  assert inspect.isfunction(lmp.util.validate.raise_if_is_file)
  assert inspect.signature(lmp.util.validate.raise_if_is_file) == Signature(
    parameters=[
      Parameter(
        annotation=str,
        default=Parameter.empty,
        kind=Parameter.KEYWORD_ONLY,
        name='path',
      ),
    ],
    return_annotation=None,
  )

  assert hasattr(lmp.util.validate, 'raise_if_wrong_ordered')
  assert inspect.isfunction(lmp.util.validate.raise_if_wrong_ordered)
  assert inspect.signature(lmp.util.validate.raise_if_wrong_ordered) == Signature(
    parameters=[
      Parameter(
        annotation=List[Union[float, int]],
        default=Parameter.empty,
        kind=Parameter.KEYWORD_ONLY,
        name='vals',
      ),
      Parameter(
        annotation=List[str],
        default=Parameter.empty,
        kind=Parameter.KEYWORD_ONLY,
        name='val_names',
      ),
    ],
    return_annotation=None,
  )
