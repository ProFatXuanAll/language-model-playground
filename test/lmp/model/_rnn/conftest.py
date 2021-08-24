r"""Setup fixture for testing :py:mod:`lmp.model._rnn.RNNModel`."""
import os
from lmp import path

import pytest

from lmp.tknzr._char import CharTknzr
from lmp.model._rnn import RNNModel


@pytest.fixture
def tknzr():
    r"""Simple CharTknzr instance"""
    return CharTknzr(
        is_uncased=True,
        max_vocab=-1,
        min_count=1,
        tk2id={
            '[bos]': 0,
            '[eos]': 1,
            '[pad]': 2,
            '[unk]': 3,
            'h': 4,
            'e': 5,
            'l': 6,
            'o': 7,
        }
    )


@pytest.fixture
def model(tknzr):
    r"""Simple RNNModel instance"""
    return RNNModel(
        d_emb=1,
        d_hid=1,
        n_hid_lyr=1,
        n_pre_hid_lyr=1,
        n_post_hid_lyr=1,
        p_emb=0.5,
        p_hid=0.5,
        tknzr=tknzr,
    )


@pytest.fixture
def cleandir(request, ckpt: int, exp_name: str) -> str:
    r"""Clean model parameters output file and directories."""
    abs_dir_path = os.path.join(path.EXP_PATH, exp_name)
    abs_file_path = os.path.join(
        abs_dir_path, RNNModel.file_name.format(ckpt)
    )

    def remove():
        if os.path.exists(abs_file_path):
            os.remove(abs_file_path)
        if os.path.exists(abs_dir_path):
            os.removedirs(abs_dir_path)

    request.addfinalizer(remove)
