r"""Tokenizer utilities."""

from typing import Dict, Optional

from lmp.tknzr import TKNZR_OPTS, BaseTknzr


def create(tknzr_name: str, **kwargs: Optional[Dict]) -> BaseTknzr:
    r"""Create new tokenizer.

    Create new tokenizer based on ``tknzr_name``.
    All keyword arguments are collected in ``**kwargs`` and are passed directly
    to tokenizer's ``__init__`` method.

    Parameters
    ==========
    tknzr_name: str
        Name of the tokenizer to create.
    kwargs: Dict, optional
        Tokenizer specific parameters.
        All tokenizer specific parameters must be passed in as keyword
        arguments.

    Returns
    =======
    lmp.tknzr.BaseTknzr
        New tokenizer instance.

    See Also
    ========
    lmp.tknzr
        All available tokenizers.

    Examples
    ========
    >>> from lmp.tknzr import WsTknzr
    >>> import lmp.util.tknzr
    >>> isinstance(lmp.util.tknzr.create('whitespace'), WsTknzr)
    True
    """
    return TKNZR_OPTS[tknzr_name](**kwargs)


def load(exp_name: str, tknzr_name: str) -> BaseTknzr:
    r"""Load pre-trained tokenizer.

    Load pre-trained tokenizer from experiment ``exp_name``.
    Tokenizer instance is load based on ``tknzr_name``.

    Parameters
    ==========
    exp_name: str
        Pre-trained tokenizer experiment name.
    tknzr_name: str
        Name of the tokenizer to load.

    Returns
    =======
    lmp.tknzr.BaseTknzr
        Pre-trained tokenizer instance.

    See Also
    ========
    lmp.tknzr
        All available tokenizers.

    Examples
    ========
    >>> from lmp.tknzr import WsTknzr
    >>> import lmp.util.tknzr
    >>> tknzr = lmp.util.tknzr.create(
    ...     tknzr_name='whitespace', is_uncased=True, max_vocab=10,
    ...     min_count=2,
    ... )
    >>> tknzr.save(exp_name='my_exp')
    >>> isinstance(
    ...     lmp.util.tknzr.load(exp_name='my_exp', tknzr_name='whitespace'),
    ...     WsTknzr,
    ... )
    True
    """
    return TKNZR_OPTS[tknzr_name].load(exp_name=exp_name)
