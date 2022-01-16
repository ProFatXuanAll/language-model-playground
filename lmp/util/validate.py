"""Checking types and values."""

import os
from typing import Any, List, Type, Union


def raise_if_empty_str(*, val: str, val_name: str) -> None:
  """Raise if ``val`` is an empty :py:class:`str`.

  Parameters
  ----------
  val: str
    Test target.
  val_name: str
    Test target name.  Mainly used to create error message.

  Raises
  ------
  ValueError
    When ``val`` is an empty :py:class:`str`.
  """
  if not val:
    raise ValueError(f'`{val_name}` must be non-empty `str`.')


def raise_if_is_directory(*, path: str) -> None:
  """Raise if ``path`` exists and is a directory.

  Parameters
  ----------
  path: str
    Test path.

  Raises
  ------
  FileExistsError
    When ``path`` exists and is a directory.
  """
  if os.path.exists(path) and os.path.isdir(path):
    raise FileExistsError(f'{path} is a directory.')


def raise_if_is_file(*, path: str) -> None:
  """Raise if ``path`` exists and is a file.

  Parameters
  ----------
  path: str
    Test path.

  Raises
  ------
  FileExistsError
    When ``path`` exists and is a file.
  """
  if os.path.exists(path) and os.path.isfile(path):
    raise FileExistsError(f'{path} is a file.')


def raise_if_not_in(*, val: Any, val_name: str, val_range: List) -> None:
  """Raise if ``val`` is not in ``val_range``.

  Parameters
  ----------
  val: Any
    Test target.
  val_name: str
    Test target name.  Mainly used to create error message.
  val_range: List
    Expected value range.

  Raises
  ------
  ValueError
    When ``val`` is not in ``val_range``.
  """
  if val not in val_range:
    raise ValueError(
      f'`{val_name}` must be one of the following values:' + ''.join(map(lambda v: f'\n- {v}', val_range))
    )


def raise_if_not_instance(*, val: Any, val_name: str, val_type: Type) -> None:
  """Raise if ``val`` is not an instance of ``val_type``.

  Parameters
  ----------
  val: Any
    Test target.
  val_name: str
    Test target name.  Mainly used to create error message.
  val_type: Type
    Expected target type.

  Raises
  ------
  TypeError
    When ``val`` is not an instance of ``val_type``.
  """
  if not isinstance(val, val_type):
    raise TypeError(f'`{val_name}` must be an instance of `{val_type.__name__}`.')


def raise_if_wrong_ordered(*, vals: List[Union[float, int]], val_names: List[str]) -> None:
  """Raise if there exist some ``i < j`` such that ``vals[i] > vals[j]``.

  Parameters
  ----------
  vals: List[Union[float, int]]
    Test targets.
  val_names: List[str]
    Test targets' names.  Mainly used to create error message.

  Raises
  ------
  ValueError
    When there exist some ``i < j`` such that ``vals[i] > vals[j]``.
  """
  for i in range(len(vals) - 1):
    if vals[i] > vals[i + 1]:
      raise ValueError(f'Must have `{" <= ".join(val_names)}`.')
