r"""Test the ability to download dataset files.

Test target:
- :py:meth:`lmp.dset.ChPoemDset.download`.
"""

import os

import lmp.path
from lmp.dset import ChPoemDset


def test_download():
    r"""Dataset must be able to download."""

    for ver in ChPoemDset.vers:
        # Download specified version.
        ChPoemDset(ver=ver).download()

        # Check file existence.
        file_path = os.path.join(
            lmp.path.DATA_PATH,
            ChPoemDset.file_name.format(ver),
        )

        assert os.path.exists(file_path)
