import abc
from typing import ClassVar, Dict, Optional

from lmp.model import BaseModel


class BaseOptim(abc.ABC):
    optim_name: ClassVar[str] = 'base'

    def __init__(
            self,
            lr: float,
            model: BaseModel,
            **kwargs: Optional[Dict],
    ):
        if not isinstance(lr, float):
            raise TypeError('`lr` must be an instance of `float`.')
        if not isinstance(model, BaseModel):
            raise TypeError('`model` must be an instance of `BaseModel`.')

    @abc.abstractmethod
    def step(self):
        raise NotImplementedError

    @staticmethod
    def train_parser(parser: argparse.ArgumentParser) -> None:
        r"""Training language model CLI arguments parser.

        Parameters
        ==========
        parser: argparse.ArgumentParser
            Parser for CLI arguments.

        See Also
        ========
        lmp.script.train_model

        Examples
        ========
        >>> import argparse
        >>> from lmp.optim import BaseOptim
        >>> parser = argparse.ArgumentParser()
        >>> BaseOptim.train_parser(parser)
        >>> args = parser.parse_args([
        ...     '--lr', '1e-4',
        ... ])
        >>> args.batch_size == 32
        True
        >>> args.lr == 1e-4
        True
        >>> args.n_epoch == 10
        True
        """
        # Required arguments.
        group = parser.add_argument_group('optimizer arguments')
        group.add_argument(
            '--batch_size',
            help='Batch size.',
            required=True,
            type=int,
        )
        group.add_argument(
            '--lr',
            help='Learning rate.',
            required=True,
            type=float,
        )
        group.add_argument(
            '--n_epoch',
            help='Number of training epoch.',
            required=True,
            type=int,
        )
