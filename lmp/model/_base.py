r"""Neural network language model base class."""

import abc
import argparse
import os
import re
from typing import ClassVar, Dict, Optional

import torch

import lmp.dset
import lmp.path


class BaseModel(abc.ABC, torch.nn.Module):
    r"""Neural network language model abstract base class.

    Provide basic functionality for save and load pred-trained model
    parameters.
    All language models must inherit :py:class:`lmp.model.BaseModel`.

    For comment throughout this class and its subclasses, we use the following
    symbols to denote the shape of tensors:

    - ``B``: Batch size.
    - ``E``: Token embedding dimension.
    - ``H``: Hidden representation dimension.
    - ``S``: Length of sequence of tokens.
    - ``V``: Vocabulary size.

    Parameters
    ==========
    kwargs: Dict, optional
        Useless parameter.
        Intently left for subclass parameters extension.

    Attributes
    ==========
    file_name: ClassVar[str]
        Model parameters output file name.
    model_name: ClassVar[str]
        Display name for model on CLI.
        Used for command line argument parsing.
        Subclass must overwrite ``model_name`` attribute.
    """
    file_name: ClassVar[str] = 'model-{}.pt'
    model_name: ClassVar[str] = 'base'

    def __init__(self, **kwargs: Optional[Dict]):
        super().__init__()

    @abc.abstractmethod
    def forward(self, batch_prev_tkids: torch.Tensor) -> torch.Tensor:
        r"""Perform forward pass.

        Parameters
        ==========
        batch_prev_tkids: torch.Tensor
            Batch of previous token ids encoded by
            :py:class:`lmp.tknzr.BaseTknzr` subclass instance.
            ``batch_prev_tkids`` has shape ``(B, S)`` and
            ``dtype == torch.int64``.

        Returns
        =======
        torch.Tensor
            Output logits after forward pass.

        Raises
        ======
        NotImplementedError
            When subclass do not implement forward pass.
        """
        raise NotImplementedError(' '.join([
            f'In class `{self.__class__.__name__}`:',
            'method `forward` not implemented yet.',
        ]))

    @abc.abstractmethod
    def loss_fn(
            self,
            batch_next_tkids: torch.Tensor,
            batch_prev_tkids: torch.Tensor,
    ) -> torch.Tensor:
        r"""Calculate language model training loss.

        Parameters
        ==========
        batch_next_tkids: torch.Tensor
            Prediction targets.
            Batch of next token ids encoded by
            :py:class:`lmp.tknzr.BaseTknzr` subclass instance.
            ``batch_next_tkids`` has same shape and ``dtype`` as
            ``batch_prev_tkids``.
        batch_prev_tkids: torch.Tensor
            Batch of previous token ids encoded by
            :py:class:`lmp.tknzr.BaseTknzr` subclass instance.
            ``batch_prev_tkids`` has shape ``(B, S)`` and
            ``dtype == torch.int64``.

        Returns
        =======
        torch.Tensor
            Average next token prediction loss.
            Returned tensor has shape ``(1)`` and ``dtype == torch.float32``.

        Raises
        ======
        NotImplementedError
            When subclass do not implement loss function.
        """
        raise NotImplementedError(' '.join([
            f'In class `{self.__class__.__name__}`:',
            'method `loss_fn` not implemented yet.',
        ]))

    @abc.abstractmethod
    def pred(self, batch_prev_tkids: torch.Tensor) -> torch.Tensor:
        r"""Next token prediction.

        Parameters
        ==========
        batch_prev_tkids: torch.Tensor
            Batch of previous token ids encoded by
            :py:class:`lmp.tknzr.BaseTknzr` subclass instance.
            ``batch_prev_tkids`` has shape ``(B, S)`` and
            ``dtype == torch.int64``.

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
        raise NotImplementedError(' '.join([
            f'In class `{self.__class__.__name__}`:',
            'method `pred` not implemented yet.',
        ]))

    @torch.no_grad()
    def ppl(
            self,
            batch_next_tkids: torch.Tensor,
            batch_prev_tkids: torch.Tensor,
    ) -> float:
        r"""Calculate mean perplexity on batch of token ids.

        Perplexity is calculate by the following formula

        .. math::

            \begin{align*}
            & \text{ppl}(w_1, w_2, \dots, w_n) \\
            &= \sqrt[n]{\frac{1}{P(w_1, w_2, \dots, w_n)}} \\
            &= \big(P(w_1, w_2, \dots, w_n)\big)^{\frac{-1}{n}} \\
            &= \big(
               P(w_1) P(w_2|w_1) \dots, P(w_n|w_1, \dots, w_{n-1}))
               \big)^{\frac{-1}{n}} \\
            &= \big(
               \prod_{i=1}^n P(w_i|w_1, \dots, w_{i-1})
               \big)^{\frac{-1}{n}} \\
            &= e^{\log\big(
               \prod_{i=1}^n P(w_i|w_1, \dots, w_{i-1})
               \big)^{\frac{-1}{n}}} \\
            &= e^{\frac{-1}{n} \sum_{i=1}^n \log P(w_i|w_1, \dots, w_{i-1})}
            \end{align*}

        Each token ids in batch will calculate their own perplexities then
        return average perplexity over batch (using arithmatic mean).

        Parameters
        ==========
        batch_next_tkids: torch.Tensor
            Prediction targets.
            Batch of next token ids encoded by
            :py:class:`lmp.tknzr.BaseTknzr` subclass instance.
            ``batch_next_tkids`` has same shape and ``dtype`` as
            ``batch_prev_tkids``.
        batch_prev_tkids: torch.Tensor
            Batch of previous token ids encoded by
            :py:class:`lmp.tknzr.BaseTknzr` subclass instance.
            ``batch_prev_tkids`` has shape ``(B, S)`` and
            ``dtype == torch.int64``.

        Returns
        =======
        float
            Average perplexity on batch of token ids.
        """
        # Get next token id's probabilities.
        # Use `batch_next_tkids` as index to gather values from prediction.
        # Since prediction has shape `(B, S, V)`, we need to gather along the
        # last dimension `V`.
        batch_next_tkids_prob = torch.gather(
            self.pred(batch_prev_tkids=batch_prev_tkids),
            -1,
            batch_next_tkids.unsqueeze(-1)
        ).squeeze(-1)

        # Calculate perplexity for each token ids sequence in batch.
        # Convert to log space for numerically save computation.
        # Exponentiate calculated result to convert back from log space.
        batch_ppl = (
            - 1 / batch_next_tkids.size(-1)
            * batch_next_tkids_prob.log().sum(dim=-1)
        ).exp()

        # Return average perplexity of the batch.
        return batch_ppl.mean().item()

    def save(self, ckpt: int, exp_name: str) -> None:
        r"""Save model parameters in compressed pickle.

        Save the trained model parameters into zip compressed pickle file and
        named it with ``self.__class__.file_name``.
        This method will create a directory for each model training experiment
        if that directory is not created before.

        Parameters
        ==========
        ckpt: int
            Model training checkpoint.
        exp_name: str
            Name of the language model training experiment.

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

        elif os.path.isdir(file_path):
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
        if not isinstance(ckpt, int):
            raise TypeError('`ckpt` must be an instance of `int`.')
        if not isinstance(exp_name, str):
            raise TypeError('`exp_name` must be an instance of `str`.')

        if ckpt < -1:
            raise ValueError('`ckpt` must satisfy `ckpt >= -1`.')
        if not exp_name:
            raise ValueError('`exp_name` must be non-empty.')

        file_dir = os.path.join(lmp.path.EXP_PATH, exp_name)
        if not os.path.exists(file_dir):
            raise FileNotFoundError(' '.join([
                f'Experiment file path {file_dir} does not exist.',
                'You must run `python -m lmp.script.train_model` first.',
            ]))

        # Load latest checkpoint.
        if ckpt == -1:
            ckpt_files = []
            for ckpt_f in os.listdir(file_dir):
                match = re.match(r'model-(\d+).pt', ckpt_f)
                if match is None:
                    continue
                ckpt_files.append(int(match.group(1)))
            ckpt = max(ckpt_files)

        # Format file name with checkpoint step.
        file_path = os.path.join(file_dir, cls.file_name.format(ckpt))

        if not os.path.exists(file_path):
            raise FileNotFoundError(' '.join([
                f'Checkpoint file path {file_path} does not exist.',
                'You must run `python -m lmp.script.train_model` first.',
            ]))

        if os.path.isdir(file_path):
            raise FileExistsError(' '.join([
                f'Checkpoint file path {file_path} is a directory.',
                f'Remove {file_path} first then do',
                '`python -m lmp.script.train_model`.',
            ]))

        # Construct new model.
        self = cls(**kwargs)

        # Load pre-trained parameters.
        self.load_state_dict(torch.load(file_path))

        return self

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
            Language model training script.

        Examples
        ========
        >>> import argparse
        >>> from lmp.model import BaseModel
        >>> parser = argparse.ArgumentParser()
        >>> BaseModel.train_parser(parser)
        >>> args = parser.parse_args([
        ...     '--batch_size', '32',
        ...     '--beta1', '0.9',
        ...     '--beta2', '0.99',
        ...     '--ckpt_step', '1000',
        ...     '--dset_name', 'wikitext-2',
        ...     '--eps', '1e-8',
        ...     '--exp_name', 'my_exp',
        ...     '--log_step', '200',
        ...     '--lr', '1e-4',
        ...     '--max_norm', '1',
        ...     '--max_seq_len', '-1',
        ...     '--n_epoch', '10',
        ...     '--tknzr_exp_name', 'my_tknzr_exp',
        ...     '--ver', 'train',
        ...     '--wd', '1e-2',
        ... ])
        >>> args.batch_size == 32
        True
        >>> args.beta1 == 0.9
        True
        >>> args.beta2 == 0.99
        True
        >>> args.ckpt_step == 1000
        True
        >>> args.dset_name == 'wikitext-2'
        True
        >>> args.eps == 1e-8
        True
        >>> args.exp_name == 'my_exp'
        True
        >>> args.log_step == 200
        True
        >>> args.lr == 1e-4
        True
        >>> args.max_norm == 1
        True
        >>> args.max_seq_len == -1
        True
        >>> args.n_epoch == 10
        True
        >>> args.seed == 42
        True
        >>> args.tknzr_exp_name == 'my_tknzr_exp'
        True
        >>> args.ver == 'train'
        True
        >>> args.wd == 1e-2
        True
        """
        # Required arguments.
        group = parser.add_argument_group('common arguments')
        group.add_argument(
            '--batch_size',
            help='Mini-batch size.',
            required=True,
            type=int,
        )
        group.add_argument(
            '--beta1',
            help='First beta coefficient of AdamW optimizer.',
            required=True,
            type=float,
        )
        group.add_argument(
            '--beta2',
            help='Second beta coefficient of AdamW optimizer.',
            required=True,
            type=float,
        )
        group.add_argument(
            '--ckpt_step',
            help='Checkpoint save interval.',
            required=True,
            type=int,
        )
        group.add_argument(
            '--dset_name',
            choices=lmp.dset.DSET_OPTS.keys(),
            help='Name of the dataset which is used to train language model.',
            required=True,
            type=str,
        )
        group.add_argument(
            '--eps',
            help='Denominator smooth term of AdamW optimizer.',
            required=True,
            type=float,
        )
        group.add_argument(
            '--exp_name',
            help='Name of the language model training experiment.',
            required=True,
            type=str,
        )
        group.add_argument(
            '--log_step',
            help='Performance log interval.',
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
            '--max_norm',
            help='Gradient max-norm constraint.',
            required=True,
            type=float,
        )
        group.add_argument(
            '--max_seq_len',
            help=' '.join([
                'Maximum sequence length constraint.',
                'Set to `-1` to allow arbitrary token sequence length.'
            ]),
            required=True,
            type=int,
        )
        group.add_argument(
            '--n_epoch',
            help='Number of training epochs.',
            required=True,
            type=int,
        )
        group.add_argument(
            '--tknzr_exp_name',
            help='Name of the pre-trained tokenizer experiment.',
            required=True,
            type=str,
        )
        group.add_argument(
            '--ver',
            help=' '.join([
                'Version of the dataset which is used to train language',
                'model.',
            ]),
            required=True,
            type=str,
        )
        group.add_argument(
            '--wd',
            help='Weight decay coefficient of AdamW optimizer.',
            required=True,
            type=float,
        )

        # Optional arguments.
        group.add_argument(
            '--seed',
            default=42,
            help='Random seed.',
            type=int,
        )
