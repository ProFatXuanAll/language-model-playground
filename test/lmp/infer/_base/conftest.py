r"""Setup fixture for testing :py:mod:`lmp.infer.BaseInfer`."""

from typing import Type

import pytest

from lmp.infer._base import BaseInfer
from lmp.model._base import BaseModel
from lmp.tknzr._base import BaseTknzr


@pytest.fixture
def subclss_infer_clss() -> Type[BaseInfer]:
    r"""Simple ``BaseInfer`` subclass."""
    class SubclssInfer(BaseInfer):
        r"""Only implement ``gen`` ."""

        def gen(self, model: BaseModel, tknzr: BaseTknzr, txt: str) -> str:
            return [txt]

    return SubclssInfer


@pytest.fixture
def subclss_infer(
        max_seq_len: int,
        subclss_infer_clss: Type[BaseInfer],
):
    r"""Simple ``BaseTknzr`` subclass instance."""
    return subclss_infer_clss(
        max_seq_len,
    )
