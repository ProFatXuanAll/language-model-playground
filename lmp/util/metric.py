"""Evaluation metrics."""

import torch

from lmp.tknzr._base import PAD_TKID


def cross_entropy_loss(batch_tkids: torch.Tensor, batch_tkids_pd: torch.Tensor) -> torch.Tensor:
  r"""Calculate cross entropy loss on batch of token ids.

  Cross entropy loss is calculated by the following formula

  .. math::

     \newcommand{\pa}[1]{\left(#1\right)}
     \begin{align*}
       \operatorname{Loss}(w_1, w_2, \dots, w_n) & = \frac{-1}{n} \sum_{i = 1}^n \log P(w_i|w_{< i})
     \end{align*}

  where :math:`n` is the length of the given token list, :math:`w_1, w_2, \dots, w_n` are tokens in the token list and
  :math:`w_{< i} = \set{w_1, w_2, \dots, w_{i - 1}}`.  Note that ``[pad]`` will not be included in cross entropy loss
  calculation results.  We do not use :py:class:`torch.nn.CrossEntropyLoss` since its reduction (summation over each
  token id) may result in NaN.

  Parameters
  ----------
  batch_tkids: torch.Tensor
    Batch of token ids which represent prediction targets.  ``batch_tkids`` has shape ``(batch_size, seq_len)`` and
    ``dtype == torch.long``.
  batch_tkids_pd: torch.Tensor
    Batch of token ids prediction probability distribution.  ``batch_tkids_pd`` has shape
    ``(batch_size, seq_len, vocab_size)`` and ``dtype == torch.float``.

  Returns
  -------
  torch.Tensor
    Cross entropy loss per sequence in the batch.  Returned tensor has shape ``(batch_size)`` and
    ``dtype == torch.float``.
  """
  # Get target token id's probabilities.  Use `batch_tkids` as indices to gather values from probability distribution.
  # Since prediction has shape `(batch_size, seq_len, vocab_size)`, we need to gather along the `vocab_size` dimension.
  # shape: (batch_size, seq_len).
  batch_tkids_p = torch.gather(input=batch_tkids_pd, dim=2, index=batch_tkids.unsqueeze(2)).squeeze(2)

  # Mask `PAD_TKID` with probability `1.0` since `log(1) = 0`.  We also mask NaN to make optimization numerically safe.
  # This is the main difference between our cross entropy loss and `torch.nn.CrossEntropyLoss`.
  mask = (batch_tkids == PAD_TKID) | torch.isnan(batch_tkids_p)
  batch_tkids_p.masked_fill_(mask=mask, value=1.0)

  # Only non-masked positions can contribute to sequence length.
  batch_seq_len = batch_tkids.size(1) - mask.sum(dim=1)

  # Calculate perplexity per sequence in batch.  Convert to log space for numerically save computation. Then convert
  # back from log space by exponentiating calculation results.
  # shape: (batch_size)
  return -batch_tkids_p.log().sum(dim=1) / batch_seq_len


def ppl(batch_tkids: torch.Tensor, batch_tkids_pd: torch.Tensor) -> torch.Tensor:
  r"""Calculate perplexity on batch of token ids.

  Perplexity is calculated by the following formula

  .. math::

     \begin{align*}
       \operatorname{Perplexity}(w_1, w_2, \dots, w_n) &= \left( P(w_1, w_2, \dots, w_n) \right)^{\frac{-1}{n}}  \\
       &= \left( P(w_1) \times P(w_2|w_1) \times \cdots \times P(w_n|w_1, \dots, w_{n-1}) \right)^{\frac{-1}{n}} \\
       &= \left( \prod_{i=1}^n P(w_i|w_{< i}) \right)^{\frac{-1}{n}}                                             \\
       &= \exp \left( \ln \left( \prod_{i=1}^n P(w_i|w_{< i}) \right)^{\frac{-1}{n}} \right)                     \\
       &= \exp \left( \frac{-1}{n} \sum_{i=1}^n \ln P(w_i|w_{< i}) \right)
     \end{align*}

  where :math:`n` is the length of the given token list and :math:`w_1, w_2, \dots, w_n` are tokens in the token list.
  Note that ``[pad]`` will not be included in perplexity calculation results.

  Parameters
  ----------
  batch_tkids: torch.Tensor
    Batch of token ids which represent prediction targets.  ``batch_tkids`` has shape ``(batch_size, seq_len)`` and
    ``dtype == torch.long``.
  batch_tkids_pd: torch.Tensor
    Batch of token ids prediction probability distribution.  ``batch_tkids_pd`` has shape
    ``(batch_size, seq_len, vocab_size)`` and ``dtype == torch.float``.

  Returns
  -------
  torch.Tensor
    Perplexity per sequence in the batch.  Returned tensor has shape ``(batch_size)`` and ``dtype == torch.float``.
  """
  return cross_entropy_loss(batch_tkids, batch_tkids_pd).exp()
