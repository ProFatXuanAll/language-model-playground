
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
        d_emb=2,
        d_hid=10,
        n_hid_lyr=10,
        n_pre_hid_lyr=10,
        n_post_hid_lyr=1,
        p_emb=0.5,
        p_hid=0.5,
        tknzr=tknzr,
    )
