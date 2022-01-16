"""Setup fixtures for testing :py:class:`lmp.dset.WikiText2Dset`."""

import os
from typing import List

import pytest

import lmp.util.path
from lmp.dset import WikiText2Dset


@pytest.fixture(scope='session')
def wiki_text_2_file_paths(request) -> List[str]:
  """Chinese poem download file path.

  After testing, clean up files and directories created during test.
  """
  abs_file_paths = [os.path.join(lmp.util.path.DATA_PATH, f'wiki.{ver}.tokens') for ver in WikiText2Dset.vers]

  def fin() -> None:
    for abs_file_path in abs_file_paths:
      if os.path.exists(abs_file_path):
        os.remove(abs_file_path)
    if os.path.exists(lmp.util.path.DATA_PATH) and not os.listdir(lmp.util.path.DATA_PATH):
      os.removedirs(lmp.util.path.DATA_PATH)

  request.addfinalizer(fin)
  return abs_file_paths
