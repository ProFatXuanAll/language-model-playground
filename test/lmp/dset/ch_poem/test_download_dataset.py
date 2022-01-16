"""Test the ability to download dataset files.

Test target:
- :py:meth:`lmp.dset.ChPoemDset.download_dataset`.
"""

import os
from typing import List

from lmp.dset import ChPoemDset


def test_download_dataset(ch_poem_file_paths: List[str]) -> None:
  """Must be able to download all chinese poems."""
  for ver in ChPoemDset.vers:
    ChPoemDset.download_dataset(ver=ver)

  assert all(map(lambda file_path: os.path.exists(file_path), ch_poem_file_paths))
