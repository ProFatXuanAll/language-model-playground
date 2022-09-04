"""Top-P inference method."""

import argparse
from typing import Any, ClassVar, List

import torch

import lmp.util.validate
from lmp.infer._base import BaseInfer
from lmp.model import BaseModel
from lmp.tknzr._base import EOS_TKID, PAD_TKID, BaseTknzr


class TopPInfer(BaseInfer):
  """Top-P inference method.

  Top-P sampling, also called nucleus sampling :footcite:`holtzman2020the`, is similar to top-K sampling but :math:`k`
  changes in each inference step.
  :math:`p` is used as **cumulative probability threshold** and :math:`k` is choosed so that the top-K highest
  probabilities have **cumulative probability less than or equal to** :math:`p`.
  Top-P sampling is a non-greedy algorithm.

  Parameters
  ----------
  max_seq_len: str, default: 32
    Maximum length constraint on generated token list.
    One can use larger contraint compare to training.
  p: float, default: 0.9
    Cumulative probability threshold.
  kwargs: typing.Any, optional
    Useless parameter.
    Intently left for subclasses inheritance.

  Attributes
  ----------
  infer_name: ClassVar[str]
    CLI name of top-P inference method is ``top-P``.
  p: float
    Cumulative probability threshold.

  See Also
  --------
  :doc:`lmp.infer </infer/index>`
    All available inference methods.
  :doc:`lmp.script.gen_txt </script/gen_txt>`
    Use pre-trained language model checkpoint to generate continual text of given text segment.
  :py:class:`~lmp.infer.TopKInfer`
    Top-K inference method.
  """

  infer_name: ClassVar[str] = 'top-P'

  def __init__(self, *, max_seq_len: int = 32, p: float = 0.9, **kwargs: Any):
    super().__init__(max_seq_len=max_seq_len)
    # `p` validation.
    lmp.util.validate.raise_if_not_instance(val=p, val_name='p', val_type=float)
    lmp.util.validate.raise_if_wrong_ordered(vals=[0.0, p, 1.0], val_names=['0.0', 'p', '1.0'])
    self.p = p

  @classmethod
  def add_CLI_args(cls, parser: argparse.ArgumentParser) -> None:
    """Add top-P inference method hyperparameters to CLI argument parser.

    Parameters
    ----------
    parser: argparse.ArgumentParser
      CLI argument parser.

    Returns
    -------
    None

    See Also
    --------
    :doc:`lmp.script.gen_txt </script/gen_txt>`
      Use pre-trained language model checkpoint to generate continual text of given text segment.

    Examples
    --------
    >>> import argparse
    >>> import math
    >>> from lmp.infer import TopPInfer
    >>> parser = argparse.ArgumentParser()
    >>> TopPInfer.infer_parser(parser)
    >>> args = parser.parse_args(['--p', '0.9'])
    >>> assert math.isclose(args.p, 0.9)
    """
    super().add_CLI_args(parser=parser)

    # Required arguments.
    group = parser.add_argument_group('top-P inference method arguments')
    group.add_argument(
      '--p',
      default=0.9,
      help='''
      Cumulative probability threshold.
      Default is ``0.9``.
      ''',
      type=float,
    )

  @torch.no_grad()
  def gen(self, model: BaseModel, tknzr: BaseTknzr, txt: str) -> str:
    """Generate continual text conditioned on given text segment.

    Top-P inference algorithm is structured as follow:

    #. Encode input text as 1 sequence batch.
    #. Remove token ids after ``<eos>`` since model is not trained to predict tokens after seeing ``<eos>``.
    #. Loop over conditional token ids to generate conditional hidden states.
    #. Loop to generate token ids.
       In each iteration, generated token id was choosed so that it is one of the top-K highest probabilities from next
       token id probability distribution, where :math:`k` is the number of token ids whose cumulative probabilities
       (probabilities are sorted in desending order) are less than or equal to ``self.p``.
       Generation loop stops when ``<eos>`` is generated or maximum length constraint is violated.
    #. Decode generated token ids into text and return.

    Parameters
    ----------
    model: ~lmp.model.BaseModel
      Pre-trained language model which will be used to generate text.
    tknzr: ~lmp.tknzr.BaseTknzr
      Pre-trained tokenizer which performs text encoding and decoding.
    txt: str
      Text segment which the generation process is conditioned on.

    Returns
    -------
    str
      Generated text.
    """
    # Get model running device.
    device = next(model.parameters()).device

    # Encode as 1 sequence batch.
    # We convert token ids to tensor and move tensor to the same running device as model.
    # shape: (1, S).
    batch_cur_tkids = torch.LongTensor([tknzr.enc(txt=txt)]).to(device)

    # Remove token ids after `<eos>` since model is not trained to predict tokens after seeing `<eos>`.
    mask = (batch_cur_tkids == EOS_TKID) | (batch_cur_tkids == PAD_TKID)
    seq_len = batch_cur_tkids.size(1) - mask.sum()
    batch_cur_tkids = batch_cur_tkids[:, :seq_len]

    # Loop over conditioned token ids to generate conditioned hidden states.
    batch_prev_states = None
    for i in range(seq_len - 1):
      _, batch_cur_states = model.pred(
        batch_cur_tkids=batch_cur_tkids[:, i].unsqueeze(1),
        batch_prev_states=batch_prev_states,
      )

      # Update hidden states.
      batch_prev_states = batch_cur_states

    # Calculate how many token at most can be generated.
    out_seq_len = self.max_seq_len - seq_len + 1

    # Generate token ids.
    # shape: (1, 1).
    batch_cur_tkids = batch_cur_tkids[:, -1].unsqueeze(1)
    gen_tkids: List[int] = []
    for _ in range(out_seq_len):
      # Get next token id prediction probability distribution.
      # shape: (1, 1, V).
      batch_next_tkids_pd, batch_cur_states = model.pred(
        batch_cur_tkids=batch_cur_tkids,
        batch_prev_states=batch_prev_states,
      )

      # Sort the probability distribution in descending order.
      # shape: (1, 1, V).
      batch_next_tkids_sort_pd, batch_next_tkids_sort = batch_next_tkids_pd.sort(dim=2, descending=True)

      # Calculate cumulative probability distribution and retrieve indices which cumulative probability are smaller
      # than threshold `self.p`.
      k = int((batch_next_tkids_sort_pd.cumsum(dim=2) <= self.p).sum().item())

      # Sometimes the highest probability is larger than `self.p`, which means model is highly confident on predicting
      # next token id.
      # Thus the above calculation will result in `k == 0`.
      # In that case we only choose the token id with the highest probability, we do this by setting `k = 1`.
      if k == 0:
        k = 1

      # The previous `k` token ids in `batch_next_tkids_sort` have cumulative probability less than or equal to
      # `self.p`.
      # We fetch them and perform further sampling.
      # shape: (1, k).
      batch_next_tkids_sort_pd = batch_next_tkids_sort_pd[..., :k]
      batch_next_tkids_sort = batch_next_tkids_sort[..., :k]

      # Reshape probability tensor to perform sampling.
      # shape: (1, k).
      batch_next_tkids_sort_pd = batch_next_tkids_sort_pd.reshape(-1, k)

      # Use the top-K highest probabilities to construct multinomial distribution.
      # Then sample token id from multinomial distribution as the next token id prediction.
      # `batch_next_tkids_topk_sample` shape: (1, 1).
      batch_next_tkids_topk_sample = torch.multinomial(batch_next_tkids_sort_pd, num_samples=1)

      # Use sampled result to fetch next token id prediction.
      # shape: (1, 1).
      batch_next_tkids = torch.gather(
        input=batch_next_tkids_sort,
        dim=2,
        index=batch_next_tkids_topk_sample.unsqueeze(2),
      ).squeeze(1)
      gen_tkid = int(batch_next_tkids[0, 0].item())
      gen_tkids.append(gen_tkid)

      # Update input token ids.
      batch_cur_tkids = batch_next_tkids

      # Update hidden states.
      batch_prev_states = batch_cur_states

      # If the prediction token id is `<eos>`, then stop generation immediately.
      if gen_tkid == EOS_TKID:
        break

    # Output generated text.
    return tknzr.dec(tkids=gen_tkids, rm_sp_tks=True)
