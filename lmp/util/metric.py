"""Evaluation metrics."""

import torch

from lmp.tknzr._base import PAD_TKID


def cross_entropy_loss(batch_tkids: torch.Tensor, batch_tkids_pd: torch.Tensor) -> torch.Tensor:
  r"""Calculate cross entropy loss on batch of token ids.

  Cross entropy loss is calculated by the following formula

  .. math::

     \newcommand{\pa}[1]{\left(#1\right)}
     \begin{align*}
       \operatorname{Loss}(x_0, x_1, \dots, x_S) & = -\sum_{i = 1}^S \log P(x_i|x_{< i})
     \end{align*}

  where :math:`S+1` is the length of the given token id list :math:`x = (x_0, x_1, \dots, x_S)`.
  The :math:`i`-th token id in :math:`x` is denote as :math:`x_i`.
  The token ids before the :math:`i`-th token id are collectively denote as :math:`x_{< i}`.
  The possible prediction results of :math:`x_{< i}` match the language model paired tokenizer's vocabulary with size
  :math:`V`, and :math:`x_i` is the correct answer among :math:`V` possible choices.
  Thus :math:`x_i \in \set{0, \dots, V-1}` for all :math:`i \in \set{0, \dots, V-1}`.
  Padding tokens ``[pad]`` will not be included in cross entropy loss calculation results.
  We do not use :py:class:`torch.nn.CrossEntropyLoss` since some context window consist entirely of ``[pad]`` tokens.
  Per token prediction loss over a mini-batch with batch size :math:`B` will be averaged.

  Parameters
  ----------
  batch_tkids: torch.Tensor
    Batch of token ids which represent prediction targets.
    ``batch_tkids`` has shape :math:`(B, S)` and ``dtype == torch.long``.
  batch_tkids_pd: torch.Tensor
    Batch of token ids prediction probability distribution.
    ``batch_tkids_pd`` has shape :math:`(B, S, V)` and ``dtype == torch.float``.

  Returns
  -------
  torch.Tensor
    Average cross entropy loss per token in the batch.
    Returned tensor has shape :math:`(1)` and ``dtype == torch.float``.
  """
  # Get target token id's probabilities.
  # Use `batch_tkids` as indices to gather values from probability distribution.
  # Since prediction has shape `(B, S, V)`, we need to gather along the `V` dimension.
  # shape: (B, S).
  batch_tkids_p = torch.gather(input=batch_tkids_pd, dim=2, index=batch_tkids.unsqueeze(2)).squeeze(2)

  # Mask `PAD_TKID` with probability `1.0` since `log(1) = 0`.
  mask = batch_tkids == PAD_TKID
  batch_tkids_p.masked_fill_(mask=mask, value=1.0)

  # Return `0.0` when all token ids in the batch are `[pad]`.
  n_non_pad = (~mask).sum()
  if n_non_pad == 0:
    return torch.tensor(0.0, requires_grad=True)

  # Calculate per token prediction loss and average over non `[pad]` token.
  # Convert to log space for numerically save computation.
  # shape: (1)
  return -batch_tkids_p.log().sum() / n_non_pad


def ppl(batch_tkids: torch.Tensor, batch_tkids_pd: torch.Tensor) -> torch.Tensor:
  r"""Calculate perplexity on batch of token ids.

  Perplexity is calculated by the following formula

  .. math::

     \begin{align*}
       \operatorname{Perplexity}(x_0, x_1, \dots, x_S) &= \pa{P(x_0, x_1, \dots, x_S)}^{\dfrac{-1}{S}}           \\
       &= \pa{P(x_1|x_0) \times P(x_2|x_0, x_1) \times \cdots \times P(x_S|x_0, \dots, x_{S-1})}^{\dfrac{-1}{S}} \\
       &= \pa{\prod_{i=1}^S P(x_i|x_{< i})}^{\dfrac{-1}{S}}                                                      \\
       &= \exp \pa{\ln \pa{\prod_{i=1}^S P(x_i|x_{< i})}^{\dfrac{-1}{S}}}                                        \\
       &= \exp \pa{\dfrac{-1}{S} \sum_{i=1}^S \ln P(x_i|x_{< i})}
     \end{align*}

  where :math:`S+1` is the length of the given token id list :math:`x = (x_0, x_1, \dots, x_S)`.
  The :math:`i`-th token id in :math:`x` is denote as :math:`x_i`.
  The token ids before the :math:`i`-th token id are collectively denote as :math:`x_{< i}`.
  The possible prediction results of :math:`x_{< i}` match the language model paired tokenizer's vocabulary with size
  :math:`V`, and :math:`x_i` is the correct answer among :math:`V` possible choices.
  Thus :math:`x_i \in \set{0, \dots, V-1}` for all :math:`i \in \set{0, \dots, V-1}`.
  Padding tokens ``[pad]`` will not be included in perplexity calculation results.

  Parameters
  ----------
  batch_tkids: torch.Tensor
    Batch of token ids which represent prediction targets.  ``batch_tkids`` has shape :math:`(B, S)` and
    ``dtype == torch.long``.
  batch_tkids_pd: torch.Tensor
    Batch of token ids prediction probability distribution.  ``batch_tkids_pd`` has shape
    :math:`(B, S, V)` and ``dtype == torch.float``.

  Returns
  -------
  torch.Tensor
    Perplexity per sequence in the batch.  Returned tensor has shape ``(B)`` and ``dtype == torch.float``.
  """
  # Get target token id's probabilities.
  # Use `batch_tkids` as indices to gather values from probability distribution.
  # Since prediction has shape `(B, S, V)`, we need to gather along the `V` dimension.
  # shape: (B, S).
  batch_tkids_p = torch.gather(input=batch_tkids_pd, dim=2, index=batch_tkids.unsqueeze(2)).squeeze(2)

  # Mask `PAD_TKID` with probability `1.0` since `log(1) = 0`.
  mask = batch_tkids == PAD_TKID
  batch_tkids_p.masked_fill_(mask=mask, value=1.0)

  # Only non-masked positions can contribute to sequence length.
  batch_seq_len = batch_tkids.size(1) - mask.sum(dim=1)

  # Return `inf` when all token ids in the batch are `[pad]`.
  if (~mask).sum() == 0:
    return torch.tensor(float('inf'))

  # Calculate perplexity per sequence in batch.
  # Convert to log space for numerically save computation and exponentiate the results back to normal range.
  # shape: (B)
  return (-batch_tkids_p.log().sum(dim=1) / batch_seq_len).exp()
