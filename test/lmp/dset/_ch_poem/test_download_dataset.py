"""Test the ability to download dataset files.

Test target:
- :py:meth:`lmp.dset._ch_poem.ChPoemDset.download_dataset`.
"""

import os

import lmp.dset._ch_poem
import lmp.vars


def test_download_dataset() -> None:
  """Must be able to download all chinese poems."""
  for ver in lmp.dset._ch_poem.ChPoemDset.vers:
    lmp.dset._ch_poem.ChPoemDset.download_dataset(ver=ver)

  assert all(
    map(
      lambda file_path: os.path.exists(file_path),
      [os.path.join(lmp.vars.DATA_PATH, f'{ver}.csv') for ver in lmp.dset._ch_poem.ChPoemDset.vers],
    )
  )
