import pytest

from lmp.tknzr._base_tknzr import BaseTknzr

class TestSign:
    r"""Test class signature."""
    # def test_max_vocab(self):
    #     max_vocab = 10
    #     tknzr = BaseTknzr(is_uncased=True, min_count=10, max_vocab=max_vocab)
    #     assert tknzr.max_vocab == max_vocab
    def test_meth(self):
        assert hasattr(BaseTknzr, '__init__')
        assert hasattr(BaseTknzr, 'tknz')
        assert hasattr(BaseTknzr, 'dtknz')
        assert hasattr(BaseTknzr, 'save')
        assert hasattr(BaseTknzr, 'load')
        assert hasattr(BaseTknzr, 'build_vocab')
        assert hasattr(BaseTknzr, 'enc')
        assert hasattr(BaseTknzr, 'dec')
        assert hasattr(BaseTknzr, 'batch_enc')
        assert hasattr(BaseTknzr, 'batch_dec')

    def test_init(self, tknzr):
        assert tknzr
