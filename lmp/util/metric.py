"""Evaluation metrics."""

import torch


@torch.no_grad()
def ppl(batch_tkids: torch.Tensor, batch_tkids_pd: torch.Tensor, eos_tkid: int, pad_tkid: int) -> torch.Tensor:
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

  Parameters
  ----------
  batch_tkids: torch.Tensor
    Batch of token ids which represent prediction targets.  ``batch_tkids`` has shape ``(B, S)`` and
    ``dtype == torch.int``.
  batch_tkids_pd: torch.Tensor
    Batch of token ids prediction probability distribution.  ``batch_tkids_pd`` has shape ``(B, S, V)`` and
    ``dtype == torch.float``.
  eos_tkid: int
    End of sentence token id which will not be included in perplexity calculation results.
  pad_tkid: int
    Padding token id which will not be included in perplexity calculation results.

  Returns
  -------
  torch.Tensor
    Perplexity per sequence in the batch.  Returned tensor has shape ``(B,)`` and ``dtype == torch.float``.
  """
  # Get target token id's probabilities.  Use `batch_tkids` as indices to gather values from probability distribution.
  # Since prediction has shape `(batch_size, seq_len, vocab_size)`, we need to gather along the `vocab_size` dimension.
  # shape: (batch_size, seq_len).
  batch_tkids_p = torch.gather(batch_tkids_pd, -1, batch_tkids.unsqueeze(-1)).squeeze(-1)

  # Mask `pad_tkid` and `eos` with probability `1.0`.  Since `log(1)` is `0` the calculation result will not get
  # affected by these tokens.
  mask = (batch_tkids == pad_tkid) | (batch_tkids == eos_tkid)
  batch_tkids_p.masked_fill_(mask=mask, value=1.0)

  # Since `pad_tkid` and `eos` are masked, only non-masked positions can contribute to sequence length.
  batch_seq_len = batch_tkids.size(1) - mask.sum(dim=1)

  # Calculate perplexity for each token ids sequence in batch.  Convert to log space for numerically save computation.
  # Exponentiate calculated result to convert back from log space.
  return (-batch_tkids_p.log().sum(dim=-1) / batch_seq_len).exp()
