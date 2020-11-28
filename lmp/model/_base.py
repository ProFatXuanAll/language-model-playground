r"""Neural network language model base class."""

import abc
import os
from typing import ClassVar, Dict, Optional

import torch

import lmp.path


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
    file_name: ClassVar[str]
        Model parameters output file name.
    model_name: ClassVar[str]
        Display name for model on CLI.
        Only used for command line argument parsing.
    """
    file_name: ClassVar[str] = 'model-{}.pt'
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

    def save(self, ckpt: int, exp_name: str) -> None:
        r"""Save model parameters in compressed pickle.

        Save the trained model parameters into zip compressed pickle file and
        named it with ``self.__class__.file_name``.
        This method will create experiment path first if experiment path does
        not exist.

        Parameters
        ==========
        ckpt: int
            Model training checkpoint.
        exp_name: str
            Training experiment name of the model.

        Raises
        ======
        FileExistsError
            When experiment directory path already exists but is not a
            directory, or when expeirment file path already exists but is a
            directory.

        See Also
        ========
        lmp.model.BaseModel.load

        Examples
        ========
        >>> from lmp.model import BaseModel
        >>> model = BaseModel()
        >>> model.save('my_exp')
        None
        """
        file_dir = os.path.join(lmp.path.EXP_PATH, exp_name)

        # Format file name with checkpoint step.
        file_path = os.path.join(
            file_dir,
            self.__class__.file_name.format(ckpt)
        )

        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        elif not os.path.isdir(file_dir):
            raise FileExistsError(f'{file_dir} is not a directory.')

        elif not os.path.isdir(file_path):
            raise FileExistsError(f'{file_path} is a directory.')

        # Save model parameters in zip compressed pickle.
        torch.save(self.state_dict(), file_path)

    @classmethod
    def load(cls, ckpt: int, exp_name: str, **kwargs: Optional[Dict]):
        r"""Load model parameters from compressed pickle.

        Load pre-trained model using saved parameters.
        Use hyperparameters (which are collected in ``**kwargs``) to construct
        new model, then load pre-trained parameters.
        Construct new model is needed since we need an exact same model
        architecture to load pre-trained parameters.
        This class method only work if pre-trained model parameters exists
        under :term:`experiment` ``exp_name``.
        Load lastest (biggest) checkpoint if ``ckpt == -1``.

        Parameters
        ==========
        ckpt: int
            Pre-trained model checkpoint.
            Load lastest (biggest) checkpoint if ``ckpt == -1``.
        exp_name: str
            Name of the existing experiment.
        kwargs: Dict, optional
            Model's hyperparameters.
            All keyword arguments are collected in ``**kwargs`` and are passed
            directly to model's ``__init__`` method.

        Raises
        ======
        FileNotFoundError
            If file ``exp/exp_name/model-ckpt.pt`` does not exist.
        TypeError
            When ``exp_name`` is not an instance of ``str``.
        ValueError
            When ``exp_name`` is empty string.

        See Also
        ========
        lmp.model.BaseModel.save

        Examples
        ========
        >>> from lmp.model import BaseModel
        >>> model = BaseModel.load('my_exp')
        """
        if not exp_name:
            raise ValueError('`exp_name` must be non-empty.')

        # Format file name with checkpoint step.
        file_path = os.path.join(
            lmp.path.EXP_PATH,
            exp_name,
            cls.file_name.format(ckpt),
        )

        if not os.path.exists(file_path):
            # TODO: add run training model script hint
            raise FileNotFoundError(f'File {file_path} does not exist.')

        if os.path.isdir(file_path):
            # TODO: add remove dir and run training model script hint
            raise FileExistsError(f'{file_path} is a directory.')

        # Construct new model.
        self = cls(**kwargs)

        # Load pre-trained parameters.
        self.load_state_dict(torch.load(file_path))

        return self
