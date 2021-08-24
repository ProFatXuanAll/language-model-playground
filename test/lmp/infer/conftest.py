r"""Setup fixture for testing :py:mod:`lmp.infer.`."""
import pytest

from lmp.tknzr._char import CharTknzr
from lmp.model._rnn import RNNModel


@pytest.fixture
def tknzr():
    r"""Simple CharTknzr instance"""
    # max_vocab size must be bigger than pad_id
    # so assign -1 as special tokens
    CharTknzr.pad_tkid = -1

    return CharTknzr(
        is_uncased=True,
        max_vocab=1,
        min_count=1,
        tk2id={
            'h': 5,
        }
    )


@pytest.fixture
def model(tknzr):
    r"""Simple RNNModel instance"""
    model = RNNModel(
        d_emb=1,
        d_hid=1,
        n_hid_lyr=1,
        n_pre_hid_lyr=1,
        n_post_hid_lyr=1,
        p_emb=0.5,
        p_hid=0.5,
        tknzr=tknzr,
    )
    return model


@pytest.fixture
def reset_pad_tkid(request):

    def reset():
        CharTknzr.pad_tkid = 2

    request.addfinalizer(reset)
