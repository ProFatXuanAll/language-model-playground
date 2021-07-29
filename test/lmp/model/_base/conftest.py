r"""Setup fixture for testing :py:mod:`lmp.model.BaseModel`."""

from typing import Type

import pytest
import torch

from lmp.model._base import BaseModel


@pytest.fixture
def subclss_model_clss() -> Type[BaseModel]:
    r"""Simple ``BaseModel`` subclass."""
    class SubclssModel(BaseModel):
        r"""Only implement ``tknz`` and ``dtknz``."""

        def forward(self, batch_prev_tkids: torch.Tensor) -> torch.Tensor:
            return ''.join(batch_prev_tkids)

        def loss_fn(
                self,
                batch_next_tkids: torch.Tensor,
                batch_prev_tkids: torch.Tensor,) -> torch.Tensor:
            return ''.join(batch_prev_tkids)

        def pred(self, batch_prev_tkids: torch.Tensor) -> torch.Tensor:
            return ''.join(batch_prev_tkids)

    return SubclssModel


@pytest.fixture
def subclss_model(
        subclss_model_clss: Type[BaseModel],
):
    r"""Simple ``BaseModel`` subclass instance."""
    return subclss_model_clss()
