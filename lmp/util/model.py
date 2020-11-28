r"""Model utilities."""

from typing import Dict, Optional

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
    >>> import lmp.util.model
    >>> isinstance(lmp.util.model.create('RNN'), RNNModel)
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
    >>> import lmp.util.model
    >>> model = lmp.util.model.create(
    ...     model_name='RNN', d_emb=10, d_hid=10, n_hid_layer=2,
    ...     n_post_hid_layer=2, n_pre_hid_layer=2, n_vocab=10, p_emb=0.1,
    ...     p_hid=0.1, pad_tkid=0,
    ... )
    >>> model.save(ckpt=0, exp_name='my_exp')
    >>> isinstance(
    ...     lmp.util.model.load(ckpt=0, exp_name='my_exp', model_name='RNN'),
    ...     RNNModel,
    ... )
    True
    """
    return MODEL_OPTS[model_name].load(ckpt=ckpt, exp_name=exp_name, **kwargs)
