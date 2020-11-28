r"""Neural network language model base class."""

import abc
from typing import ClassVar, Dict, Optional

import torch
import torch.nn as nn


class BaseModel(abc.ABC, torch.nn.Module):
    r"""Neural network language model base class.

    Provide basic functionality for save and load pred-trained model
    parameters.
    All language model must inherit :py:class:`lmp.model.BaseModel`.

    Parameters
    ==========
    kwargs: Dict, optional
        Useless parameter.
        Intended left for subclass parameters extension.

    Attributes
    ==========
    model_name: ClassVar[str]
        Display name for model on CLI.
        Only used for command line argument parsing.
    """
    model_name: ClassVar[str] = 'base'

    def __init__(self, **kwargs: Optional[Dict]):
        super().__init__()

    @abc.abstractmethod
    def forward(self, batch_tkid: torch.Tensor) -> torch.Tensor:
        r"""Perform forward pass.

        Parameters
        ==========
        batch_tkid: torch.Tensor
            Batch of token ids encoded by :py:class:`lmp.tknzr.BaseTknzr`.
            ``batch_tkid`` has shape ``(B, S)`` and ``dtype == torch.int64``.

        Returns
        =======
        torch.Tensor
            Output logits after forward pass.

        Raises
        ======
        NotImplementedError
            When subclass do not implement forward pass.
        """
        raise NotImplementedError(
            f'In class `{self.__class__.__name__}`: '
            'method `forward` not implemented yet.'
        )

    @abc.abstractmethod
    def cal_loss(
            self,
            batch_tkid: torch.Tensor,
            batch_next_tkid: torch.Tensor
    ) -> torch.Tensor:
        r"""Calculate language model training loss.

        Parameters
        ==========
        batch_tkid: torch.Tensor
            Batch of token ids encoded by :py:class:`lmp.tknzr.BaseTknzr`.
            ``batch_tkid`` has shape ``(B, S)`` and ``dtype == torch.int64``.
        batch_next_tkid: torch.Tensor
            Prediction targets.
            Batch of token ids encoded by :py:class:`lmp.tknzr.BaseTknzr`.
            ``batch_next_tkid`` has same shape and ``dtype`` as ``batch_tkid``.

        Returns
        =======
        torch.Tensor
            Average next token prediction loss.

        Raises
        ======
        NotImplementedError
            When subclass do not implement loss function.
        """
        raise NotImplementedError(
            f'In class `{self.__class__.__name__}`: '
            'method `cal_loss` not implemented yet.'
        )

    @abc.abstractmethod
    def pred(self, batch_tkid: torch.Tensor) -> torch.Tensor:
        r"""Next token prediction.

        Parameters
        ==========
        batch_tkid: torch.Tensor
            Batch of token ids encoded by :py:class:`lmp.tknzr.BaseTknzr`.
            ``batch_tkid`` has shape ``(B, S)`` and ``dtype == torch.int64``.

        Returns
        =======
        torch.Tensor
            Predicition for next token.
            Return tensor has shape ``(B, S, V)`` and
            ``dtype == torch.float32``.

        Raises
        ======
        NotImplementedError
            When subclass do not implement next token prediction.
        """
        raise NotImplementedError(
            f'In class `{self.__class__.__name__}`: '
            'method `pred` not implemented yet.'
        )
