r"""Setup fixture for testing :py:mod:`lmp.tknzr.BaseTknzr`."""

from typing import Dict, List, Type, Union

import pytest

from lmp.tknzr._base import BaseTknzr


@pytest.fixture
def subclss_tknzr_clss() -> Type[BaseTknzr]:
    r"""Simple ``BaseTknzr`` subclass."""
    class SubclssTknzr(BaseTknzr):
        r"""Only implement ``tknz`` and ``dtknz``."""

        def tknz(self, txt: str) -> List[str]:
            return txt.split(' ')

        def dtknz(self, tks: List[str]) -> str:
            return ''.join(tks)

    return SubclssTknzr


@pytest.fixture
def subclss_tknzr(
        is_uncased: bool,
        max_vocab: int,
        min_count: int,
        subclss_tknzr_clss: Type[BaseTknzr],
        tk2id: Union[None, Dict[str, int]],
):
    r"""Simple ``BaseTknzr`` subclass instance."""
    return subclss_tknzr_clss(
        is_uncased,
        max_vocab,
        min_count,
        tk2id=tk2id,
    )
