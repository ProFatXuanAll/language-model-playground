"""Test the ability to download dataset files.

Test target:
- :py:meth:`lmp.dset.WikiText2Dset.download_dataset`.
"""

import os
from typing import List

from lmp.dset import WikiText2Dset


def test_download_dataset(wiki_text_2_file_paths: List[str]) -> None:
  """Must be able to download all chinese poems."""
  WikiText2Dset.download_dataset()

  assert all(map(lambda file_path: os.path.exists(file_path), wiki_text_2_file_paths))
