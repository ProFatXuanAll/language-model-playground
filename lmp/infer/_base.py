r"""Inference method base class."""
import abc
import argparse
from typing import ClassVar, Dict, Optional

import torch

from lmp.model import BaseModel
from lmp.tknzr import BaseTknzr


class BaseInfer(abc.ABC):
    r"""Inference method abstract base class.

    All inference methods must inherit :py:class:`lmp.infer.BaseInfer`.

    For comment throughout this class and its subclasses, we use the following
    symbols to denote the shape of tensors:

    - ``B``: Batch size.
    - ``S'``: Length of original sequence of tokens.
    - ``S``: Length of truncated sequence of tokens.
    - ``V``: Vocabulary size.

    Parameters
    ==========
    kwargs: Dict, optional
        Useless parameter.
        Intently left for subclass parameters extension.
    max_seq_len: str
        Generated sequence of tokens maximum sequence length constraint.
        Must satisfy ``0 <= max_seq_len <= BaseInfer.hard_max_seq_len``.
        If constraint is violated, then replace ``max_seq_len`` with
        ``BaseInfer.hard_max_seq_len``.

    Attributes
    ==========
    hard_max_seq_len: ClassVar[int]
        Hard limit of maximum sequence length.
        This is set to avoid generating too many tokens.
    infer_name: ClassVar[str]
        Display name for inference method on CLI.
        Used for command line argument parsing.
        Subclass must overwrite ``infer_name`` attribute.
    max_seq_len: str
        Maximum sequence length constraint of generated sequence of tokens.

    Raises
    ======
    TypeError
        If ``max_seq_len`` is not an instance of :py:class:`int`.
    """
    hard_max_seq_len: ClassVar[int] = 512
    infer_name: ClassVar[str] = 'base'

    def __init__(self, max_seq_len: int, **kwargs: Optional[Dict]):
        if not isinstance(max_seq_len, int):
            raise TypeError('`max_seq_len` must be an instance of `int`.')

        # Set `self.max_seq_len` to `self.__class__.hard_max_seq_len` if
        # violate maximum sequence length constraint.
        if not (0 <= max_seq_len <= self.__class__.hard_max_seq_len):
            self.max_seq_len = self.__class__.hard_max_seq_len
        # Use `max_seq_len` normally.
        else:
            self.max_seq_len = max_seq_len

    @torch.no_grad()
    @abc.abstractmethod
    def gen(
            self,
            model: BaseModel,
            tknzr: BaseTknzr,
            txt: str,
    ) -> str:
        r"""Generate text conditional on text segment.

        Parameters
        ==========
        model: lmp.model.BaseModel
            Pre-trained language model to generate text.
        tknzr: lmp.tknzr.BaseTknzr
            Pre-trained tokenizer for text segment encoding.
        txt: str
            Text segment to condition on.

        Returns
        =======
        str
            Generated text.

        Raises
        ======
        NotImplementedError
            When subclass do not implement text generation.
        """
        raise NotImplementedError(' '.join([
            f'In class `{self.__class__.__name__}`:',
            'method `gen` not implemented yet.',
        ]))

    @staticmethod
    def infer_parser(parser: argparse.ArgumentParser) -> None:
        r"""Language model text generation CLI arguments parser.

        Parameters
        ==========
        parser: argparse.ArgumentParser
            Parser for CLI arguments.

        See Also
        ========
        lmp.script.generate_text
            Generate text using pre-trained language model.

        Examples
        ========
        >>> import argparse
        >>> from lmp.infer import BaseInfer
        >>> parser = argparse.ArgumentParser()
        >>> BaseInfer.infer_parser(parser)
        >>> args = parser.parse_args([
        ...     '--ckpt', '5000',
        ...     '--exp_name', 'my_exp',
        ...     '--txt', 'Hello world',
        ... ])
        >>> args.ckpt == 5000
        True
        >>> args.exp_name == 'my_exp'
        True
        >>> args.txt == 'Hello world'
        True
        >>> args.seed == 42
        True
        """
        # Required arguments.
        group = parser.add_argument_group('common arguments')
        group.add_argument(
            '--ckpt',
            help='Pre-trained language model checkpoint to inference.',
            required=True,
            type=int,
        )
        group.add_argument(
            '--exp_name',
            help='Pre-trained language model experiment name.',
            required=True,
            type=str,
        )
        group.add_argument(
            '--txt',
            help='Text segment to conditional on.',
            required=True,
            type=str,
        )

        # Optional arguments.
        group.add_argument(
            '--seed',
            default=42,
            help='Random seed.',
            type=int,
        )
