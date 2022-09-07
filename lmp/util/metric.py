"""Evaluation metrics."""

import torch

from lmp.vars import PAD_TKID


def nll(batch_tkids: torch.Tensor, batch_tkids_pd: torch.Tensor, use_log2: bool = True) -> torch.Tensor:
  r"""Calculate negative log-likelihood :math:`-\log p` on batch token ids.

  Let :math:`x = \pa{x^1, \dots, x^B}` be a batch of token sequences.
  Suppose that each token sequence has length :math:`S+1` and each token is defined in a vocabulary with size :math:`V`.
  Let :math:`x^b = \pa{x_1^b, \dots, x_{S+1}^b}` be the :math:`b`-th token sequence in the batch :math:`x`.
  Suppose that the probability :math:`\Pr\pa{x_{t+1}^b \vert x_1^b, \dots, x_t^b}` of the next token :math:`x_{t+1}^b`
  when seeing previous tokens :math:`x_1^b, \dots, x_t^b` is given.
  Let :math:`R` be the returned tensor with shape :math:`(B, S)`.
  Then the :math:`b`-th row, :math:`t`-th column of the returned tensor :math:`R` is defined as follow:

  .. math::

    R_{b,t} = -\log_2 \Pr\pa{x_{t+1} \vert x_1^b, \dots,  x_t^b}.

  If :math:`x_{t+1}^b` is padding token, then we assign :math:`R_{b,t}` to zero.

  Parameters
  ----------
  batch_tkids: torch.Tensor
    Batch of token ids which represent prediction targets.
    ``batch_tkids`` has shape :math:`(B, S)` and ``dtype == torch.long``.
  batch_tkids_pd: torch.Tensor
    Batch of token id prediction probability distributions.
    ``batch_tkids_pd`` has shape :math:`(B, S, V)` and ``dtype == torch.float``.
  use_log2: bool, default: True
    Set to ``True`` to use :math:`\log_2`.
    Set to ``False`` to use :math:`\ln`.

  Returns
  -------
  torch.Tensor
    :math:`-\log p` tensor.
    Returned tensor has shape :math:`(B, S)` and ``dtype == torch.float``.
  """
  # Get target token id's probabilities.
  # Use `batch_tkids` as indices to gather values from probability distribution.
  # Since prediction has shape `(B, S, V)`, we need to gather along the `V` dimension.
  # shape: (B, S).
  batch_tkids_p = torch.gather(input=batch_tkids_pd, dim=2, index=batch_tkids.unsqueeze(2)).squeeze(2)

  # Mask `PAD_TKID`.
  mask = batch_tkids != PAD_TKID

  if use_log2:
    return mask * -batch_tkids_p.log2()
  return mask * -batch_tkids_p.log()
