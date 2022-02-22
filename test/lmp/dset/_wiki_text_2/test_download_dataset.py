"""Test the ability to download dataset files.

Test target:
- :py:meth:`lmp.dset._wiki_text_2.WikiText2Dset.download_dataset`.
"""

import os

import lmp.dset._wiki_text_2
import lmp.util.path


def test_download_dataset() -> None:
  """Must be able to download all wiki text."""
  lmp.dset._wiki_text_2.WikiText2Dset.download_dataset()

  assert all(
    map(
      lambda file_path: os.path.exists(file_path),
      [os.path.join(lmp.util.path.DATA_PATH, f'wiki.{ver}.tokens') for ver in lmp.dset._wiki_text_2.WikiText2Dset.vers],
    )
  )
