r"""Setup fixtures for testing :py:mod:`lmp.model.BaseModel`."""

from typing import Type

import pytest
import torch
import torch.nn.functional as F

from lmp.model import BaseModel


@pytest.fixture
def subclss_model_clss() -> Type[BaseModel]:
    r"""Simple ``BaseModel`` subclass."""
    class SubclssModel(BaseModel):
        r"""Only implement ``forward`` and ``loss_fn`` and ``pred``."""

        def forward(self, batch_prev_tkids: torch.Tensor) -> torch.Tensor:
            return torch.ones(
                batch_prev_tkids.size(0),
                batch_prev_tkids.size(1),
                batch_prev_tkids.max() + 1,
            )

        def loss_fn(
            self,
            batch_next_tkids: torch.Tensor,
            batch_prev_tkids: torch.Tensor,
        ) -> torch.Tensor:
            return torch.ones(1)

        def pred(self, batch_prev_tkids: torch.Tensor) -> torch.Tensor:
            return F.softmax(self(batch_prev_tkids), dim=-1)

    return SubclssModel


@pytest.fixture
def subclss_model(
        subclss_model_clss: Type[BaseModel],
):
    r"""Simple ``BaseModel`` subclass instance."""
    return subclss_model_clss()
