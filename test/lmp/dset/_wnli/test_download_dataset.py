"""Test the ability to download dataset files.

Test target:
- :py:meth:`lmp.dset._wnli.WNLIDset.download_dataset`.
"""

import os

import lmp.dset._wnli
import lmp.vars


def test_download_dataset() -> None:
  """Must be able to download all wiki text."""
  lmp.dset._wnli.WNLIDset.download_dataset()

  assert all(
    map(
      lambda file_path: os.path.exists(file_path),
      [os.path.join(lmp.vars.DATA_PATH, f'wnli.{ver}.tsv') for ver in lmp.dset._wnli.WNLIDset.vers],
    )
  )
