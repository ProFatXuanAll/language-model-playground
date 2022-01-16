"""Setup fixtures for testing :py:class:`lmp.dset.ChPoemDset`."""

import os
from typing import List

import pytest

import lmp.util.path
from lmp.dset import ChPoemDset


@pytest.fixture(scope='session')
def ch_poem_file_paths(request) -> List[str]:
  """Chinese poem download file path.

  After testing, clean up files and directories created during test.
  """
  abs_file_paths = [os.path.join(lmp.util.path.DATA_PATH, f'{ver}.csv') for ver in ChPoemDset.vers]

  def fin() -> None:
    for abs_file_path in abs_file_paths:
      if os.path.exists(abs_file_path):
        os.remove(abs_file_path)
    if os.path.exists(lmp.util.path.DATA_PATH) and not os.listdir(lmp.util.path.DATA_PATH):
      os.removedirs(lmp.util.path.DATA_PATH)

  request.addfinalizer(fin)
  return abs_file_paths
