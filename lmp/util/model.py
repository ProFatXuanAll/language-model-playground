r"""Model utilities."""

import os
import re
from typing import Dict, List, Optional

import lmp.path
from lmp.model import MODEL_OPTS, BaseModel


def create(model_name: str, **kwargs: Optional[Dict]) -> BaseModel:
    r"""Create new language model.

    Create new language model based on ``model_name``.
    All keyword arguments are collected in ``**kwargs`` and are passed directly
    to model's ``__init__`` method.

    Parameters
    ==========
    model_name: str
        Name of the lanugage model to create.
    kwargs: Dict, optional
        Model's specific hyperparameters.
        All model specific hyperparameters must be passed in as keyword
        arguments.

    Returns
    =======
    lmp.model.BaseModel
        New language model instance.

    See Also
    ========
    lmp.model
        All available language models.

    Examples
    ========
    >>> from lmp.model import RNNModel
    >>> from lmp.tknzr import CharTknzr
    >>> import lmp.util.model
    >>> tknzr = CharTknzr(is_uncased=False, max_vocab=10, min_count=2)
    >>> model = lmp.util.model.create(
    ...     model_name='RNN', d_emb=10, d_hid=10, n_hid_lyr=2,
    ...     n_post_hid_lyr=2, n_pre_hid_lyr=2, tknzr=tknzr, p_emb=0.1,
    ...     p_hid=0.1,
    ... )
    >>> isinstance(model, RNNModel)
    True
    """
    return MODEL_OPTS[model_name](**kwargs)


def load(
        ckpt: int,
        exp_name: str,
        model_name: str,
        **kwargs: Optional[Dict],
) -> BaseModel:
    r"""Load pre-trained language model.

    Load pre-trained language model from experiment ``exp_name``.
    Language model instance is load based on ``model_name`` and ``ckpt``.

    Parameters
    ==========
    ckpt: int

    exp_name: str
        Pre-trained language model experiment name.
    model_name: str
        Name of the language model to load.
    kwargs: Dict, optional
        Model's specific hyperparameters.
        All model specific hyperparameters must be passed in as keyword
        arguments.

    Returns
    =======
    lmp.model.BaseModel
        Pre-trained language model instance.

    See Also
    ========
    lmp.model
        All available models.

    Examples
    ========
    >>> from lmp.model import RNNModel
    >>> from lmp.tknzr import CharTknzr
    >>> import lmp.util.model
    >>> tknzr = CharTknzr(is_uncased=False, max_vocab=10, min_count=2)
    >>> model = lmp.util.model.create(
    ...     model_name='RNN', d_emb=10, d_hid=10, n_hid_lyr=2,
    ...     n_post_hid_lyr=2, n_pre_hid_lyr=2, tknzr=tknzr, p_emb=0.1,
    ...     p_hid=0.1,
    ... )
    >>> model.save(ckpt=1, exp_name='my_exp')
    >>> load_model = lmp.util.model.load(
    ...     ckpt=1, exp_name='my_exp', model_name='RNN', d_emb=10, d_hid=10,
    ...     n_hid_lyr=2, n_post_hid_lyr=2, n_pre_hid_lyr=2, tknzr=tknzr,
    ...     p_emb=0.1, p_hid=0.1,
    ... )
    >>> isinstance(load_model, RNNModel)
    True
    >>> (model.emb.weight == load_model.emb.weight).all().item()
    True
    """
    return MODEL_OPTS[model_name].load(ckpt=ckpt, exp_name=exp_name, **kwargs)


def list_ckpts(
        exp_name: str,
        first_ckpt: int,
        last_ckpt: int,
) -> List[int]:
    r"""List all pre-trained model checkpoint.

    List all pre-trained model checkpoint from ``first_ckpt`` to ``last_ckpt``.
    Last checkpoint is also included.
    All checkpoint must named with format ``r'model-\d+.pt'``.
    If ``first_ckpt == -1``, then only include last checkpoint.
    If ``last_ckpt == -1``, then include all checkpoint start after
    ``first_ckpt``.

    Parameters
    ==========
    exp_name: str
        Pre-trained language model experiment name.
    first_ckpt: int
        First checkpoint to include.
    last_ckpt: int
        Last checkpoint to include.

    Returns
    =======
    List[int]
        All available checkpoints from ``first_ckpt`` to ``last_ckpt``.
        Checkpoint are sorted in ascending order.

    Raises
    ======
    ValueError
        If no checkpoint is available between ``first_ckpt`` to ``last_ckpt``.
    """
    ckpts = []
    for ckpt in os.listdir(os.path.join(lmp.path.EXP_PATH, exp_name)):
        match = re.match(r'model-(\d+).pt', ckpt)
        if match is None:
            continue
        ckpts.append(int(match.group(1)))

    if first_ckpt == -1:
        first_ckpt = max(ckpts)
    if last_ckpt == -1:
        last_ckpt = max(ckpts)

    ckpts = list(filter(lambda ckpt: first_ckpt <= ckpt <= last_ckpt, ckpts))

    if not ckpts:
        raise ValueError(' '.join([
            f'No checkpoint available between {first_ckpt} to {last_ckpt}.',
            'You must run `python -m lmp.script.train_model` first.',
        ]))

    return sorted(ckpts)
