r"""Setup fixture for testing :py:mod:`lmp.tknzr.BaseTknzr`."""

from typing import Dict, List, Type, Union

import pytest

from lmp.tknzr._base_tknzr import BaseTknzr


@pytest.fixture
def subclass_tknzr_cstr() -> Type[BaseTknzr]:
    r"""Simple ``BaseTknzr`` subclass."""
    class SubclassTknzr(BaseTknzr):
        r"""Only implement ``tknz`` and ``dtknz``."""

        def tknz(self, seq: str) -> List[str]:
            return [seq]

        def dtknz(self, tks: List[str]) -> str:
            return ''.join(tks)

    return SubclassTknzr


@pytest.fixture
def subclass_tknzr(
        is_uncased: bool,
        max_vocab: int,
        min_count: int,
        subclass_tknzr_cstr: Type[BaseTknzr],
        tk2id: Union[None, Dict[str, int]],
):
    r"""Simple ``BaseTknzr`` subclass instance."""
    return subclass_tknzr_cstr(
        is_uncased,
        max_vocab,
        min_count,
        tk2id=tk2id,
    )
