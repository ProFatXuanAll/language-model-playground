r"""Test the downloaded file

Test target:
- :py:meth:`lmp.tknzr._wiki_text_2.WikiText2Dset.download`.
"""

import os
import pandas as pd
from zipfile import ZipFile

import pytest 

from lmp.dset._wiki_text_2 import WikiText2Dset
from lmp import path


def test_dset_file_exist():
    r"""Dataset must be downloaded to right places"""

    wi_dset = WikiText2Dset()

    wi_dset.download()

    file_name = wi_dset.file_name.format(wi_dset.ver)
    file_path = os.path.join(path.DATA_PATH, file_name)

    assert os.path.exists(file_path) == True
