r"""Test the ability to download dataset files.

Test target:
- :py:meth:`lmp.dset.WikiText2Dset.download`.
"""

import os

import lmp.path
from lmp.dset import WikiText2Dset


def test_download():
    r"""Dataset must be able to download."""

    for ver in WikiText2Dset.vers:
        # Download specified version.
        WikiText2Dset(ver=ver).download()

        # Check file existence.
        file_path = os.path.join(
            lmp.path.DATA_PATH,
            WikiText2Dset.file_name.format(ver),
        )

        assert os.path.exists(file_path)
