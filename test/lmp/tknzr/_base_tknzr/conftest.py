from typing import List
import pytest
from lmp.tknzr._base_tknzr import BaseTknzr

@pytest.fixture(params=[
    {'is_uncased': True, 'min_count': 1, 'max_vocab': 1},
    {'is_uncased': True, 'min_count': 2, 'max_vocab': 1},
    {'is_uncased': True, 'min_count': 1, 'max_vocab': 2},
    {'is_uncased': True, 'min_count': 2, 'max_vocab': 2},
])
def tknzr(request):
    class SomeTknzr(BaseTknzr):
        def tknz(self, seq: str) -> List[str]:
            return [seq]

        def dtknz(self, tks: List[str]) -> str:
            return ''.join(tks)

    return SomeTknzr(*request.param)
