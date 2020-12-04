r"""Inference method utilities."""

from typing import Dict, Optional

from lmp.infer import INFER_OPTS, BaseInfer


def create(infer_name: str, **kwargs: Optional[Dict]) -> BaseInfer:
    r"""Create inference method instance.

    Create inference method instance based on ``infer_name``.
    All keyword arguments are collected in ``**kwargs`` and are passed directly
    to inference method's ``__init__`` method.

    Parameters
    ==========
    infer_name: str
        Name of the inference method to create.
    kwargs: Dict, optional
        Inference method specific parameters.
        All inference method specific parameters must be passed in as keyword
        arguments.

    Returns
    =======
    lmp.infer.BaseInfer
        Inference method instance.

    See Also
    ========
    lmp.infer
        All available inference methods.

    Examples
    ========
    >>> from lmp.infer import Top1Infer
    >>> import lmp.util.infer
    >>> isinstance(lmp.util.infer.create('top-1'), Top1Infer)
    True
    """
    return INFER_OPTS[infer_name](**kwargs)
