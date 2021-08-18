r"""Test the downloaded file

Test target:
- :py:meth:`lmp.tknzr._wiki_text_2.WikiText2Dset.download`.
"""
import os


from lmp.dset._wiki_text_2 import WikiText2Dset
from lmp import path
from test.lmp.dset._wiki_text_2.conftest import cleandir


def test_dset_file_exist(dset_ver):
    r"""Dataset must be downloaded to right places"""

    wi_dset = WikiText2Dset()

    assert os.path.exists(path.DATA_PATH)
    assert os.path.exists(wi_dset.file_path)

    cleandir(dset_ver)
