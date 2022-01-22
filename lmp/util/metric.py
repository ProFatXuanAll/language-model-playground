"""Evaluation metrics."""

import torch


@torch.no_grad()
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

  Parameters
  ----------
  batch_tkids: torch.Tensor
    Batch of token ids which represent prediction targets.  ``batch_tkids`` has shape ``(B, S)`` and
    ``dtype == torch.int``.
  batch_tkids_pd: torch.Tensor
    Batch of token ids prediction probability distribution.  ``batch_tkids_pd`` has the same shape but with
    ``dtype == torch.float``.

  Returns
  -------
  torch.Tensor
    Perplexity per sequence in the batch.  Returned tensor has shape ``(B,)`` and ``dtype == torch.float``.
  """
  # Get target token id's probabilities.  Use `batch_tkids` as indices to gather values from probability distribution.
  # Since prediction has shape `(B, S, V)`, we need to gather along the last dimension `V`.
  batch_tkids_p = torch.gather(batch_tkids_pd, -1, batch_tkids.unsqueeze(-1)).squeeze(-1)

  # Calculate perplexity for each token ids sequence in batch.  Convert to log space for numerically save computation.
  # Exponentiate calculated result to convert back from log space.
  return (-1 / batch_tkids.size(-1) * batch_tkids_p.log().sum(dim=-1)).exp()
