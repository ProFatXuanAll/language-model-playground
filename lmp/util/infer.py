"""Inference method utilities."""

from typing import Any

import lmp.util.validate
from lmp.infer import INFER_OPTS, BaseInfer


def create(infer_name: str, **kwargs: Any) -> BaseInfer:
  """Create inference method instance by inference method's name.

  Inference method's arguments are collected in ``**kwargs`` and are passed directly to inference method's constructor.

  Parameters
  ----------
  infer_name: str
    Name of the inference method to create.
  kwargs: typing.Any, optional
    Inference method's parameters.

  Returns
  -------
  lmp.infer.BaseInfer
    Inference method instance.

  See Also
  --------
  lmp.infer
    All available inference methods.

  Examples
  --------
  >>> from lmp.infer import TopKInfer
  >>> import lmp.util.infer
  >>> isinstance(lmp.util.infer.create(infer_name=TopKInfer.infer_name, k=5), TopKInfer)
  True
  """
  # `infer_name` validation.
  lmp.util.validate.raise_if_not_instance(val=infer_name, val_name='infer_name', val_type=str)
  lmp.util.validate.raise_if_not_in(val=infer_name, val_name='infer_name', val_range=list(INFER_OPTS.keys()))

  # `kwargs` validation will be performed in `BaseInfer.__init__`.
  # Currently `mypy` cannot perform static type check on `**kwargs`, and I think it can only be check by runtime and
  # therefore `mypy` may no be able to solve this issue forever.  So we use `# type: ignore` to silence error.
  return INFER_OPTS[infer_name](**kwargs)  # type: ignore
