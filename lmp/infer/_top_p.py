"""Top-p inference method."""

import argparse
from typing import Any, ClassVar, List

import torch

import lmp.util.validate
from lmp.infer._base import BaseInfer
from lmp.model import BaseModel
from lmp.tknzr import BaseTknzr


class TopPInfer(BaseInfer):
  """Top-p inference method.

  Top-p sampling, also called nucleus sampling [1]_, is similar to top-k sampling but :math:`k` changes during each
  inference step.  :math:`p` is used as **cumulative probability threshold** and :math:`k` is choosed in so that the
  top-k highest probabilities have **cumulative probability less than or equal to** :math:`p`.  top-p sampling is a
  non-greedy algorithm.

  Parameters
  ----------
  max_seq_len: str
    Maximum length constraint on generated token list.  One can use larger contraint compare to training.
  p: float
    Cumulative probability threshold.
  kwargs: typing.Any, optional
    Useless parameter.  Intently left for subclasses inheritance.

  Attributes
  ----------
  infer_name: ClassVar[str]
    CLI name of top-p inference method is ``top-p``.
  p: float
    Cumulative probability threshold.

  See Also
  --------
  :doc:`lmp.infer </infer/index>`
    All available inference methods.
  lmp.infer.TopKInfer
    Top-k inference method.
  lmp.script.gen_txt
    Use pre-trained language model checkpoint to generate continual text of given text segment.

  References
  ----------
  .. [1] Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2019, September).
     `The Curious Case of Neural Text Degeneration`_. In International Conference on Learning Representations.

  .. _`The Curious Case of Neural Text Degeneration`: https://openreview.net/forum?id=rygGQyrFvH
  """

  infer_name: ClassVar[str] = 'top-p'

  def __init__(self, max_seq_len: int, p: float, **kwargs: Any):
    super().__init__(max_seq_len=max_seq_len)
    # `p` validation.
    lmp.util.validate.raise_if_not_instance(val=p, val_name='p', val_type=float)
    lmp.util.validate.raise_if_wrong_ordered(vals=[0.0, p, 1.0], val_names=['0.0', 'p', '1.0'])
    self.p = p

  @torch.no_grad()
  def gen(self, model: BaseModel, tknzr: BaseTknzr, txt: str) -> str:
    """Generate continual text conditioned on given text segment.

    Top-p inference algorithm is structured as follow:

    #. Encode input text as 1 sample batch.
    #. Remove token ids after ``[eos]`` since model is not trained to predict tokens after seeing ``[eos]``.
    #. Loop over conditioned token ids to generate conditioned hidden states.
    #. Loop to generate token ids.  In each iteration, generated token id was choosed so that it is one of the top-k
       highest probabilities from next token id prediction probability distribution, where :math:`k` is the number of
       token ids whose cumulative probability (after sorting probability in desending order) is less than or equal to
       ``self.p``.  Generating loop will stop early if ``[eos]`` is generated, otherwise generating loop only stop when
       maximum length constraint enforced by ``self.max_seq_len`` is violated.
    #. Decode generated token ids into text and return.

    Parameters
    ----------
    model: lmp.model.BaseModel
      Pre-trained language model which will be used to generate text.
    tknzr: lmp.tknzr.BaseTknzr
      Pre-trained tokenizer which perform text encoding and decoding.
    txt: str
      Text segment which the generation process is conditioned on.

    Returns
    -------
    str
      Generated text.
    """
    # Get model running device.
    device = next(model.parameters()).device

    # Encode as 1 sample batch.  We convert token ids to tensor and move tensor to the same running device as model.
    # shape: (1, max_seq_len).
    batch_cur_tkids = torch.LongTensor(tknzr.batch_enc(batch_txt=[txt], max_seq_len=self.max_seq_len)).to(device)

    # Remove token ids after `[eos]` since model is not trained to predict tokens after seeing `[eos]`.
    mask = (batch_cur_tkids == tknzr.eos_tkid) | (batch_cur_tkids == tknzr.pad_tkid)
    seq_len = batch_cur_tkids.size(1) - mask.sum()
    batch_cur_tkids = batch_cur_tkids[:, :seq_len]

    # Loop over conditioned token ids to generate conditioned hidden states.
    batch_prev_states = None
    for i in range(seq_len - 1):
      _, batch_prev_states = model.pred(batch_cur_tkids=batch_cur_tkids[:, i], batch_prev_states=batch_prev_states)

    # Calculate how many token at most can be generated.
    out_seq_len = self.max_seq_len - seq_len + 1

    # Generate token ids.
    batch_cur_tkids = batch_cur_tkids[:, -1]
    gen_tkids: List[int] = []
    for _ in range(out_seq_len):
      # Get next token id prediction probability distribution.
      # shape: (1, vocab_size)
      batch_next_tkids_pd, batch_prev_states = model.pred(
        batch_cur_tkids=batch_cur_tkids,
        batch_prev_states=batch_prev_states,
      )

      # Sort the probability distribution in descending order.
      # shape: (1, vocab_size).
      batch_next_tkids_sort_pd, batch_next_tkids_sort = batch_next_tkids_pd.sort(dim=1, descending=True)

      # Calculate cumulative probability distribution and retrieve indices which cumulative probability are smaller
      # than threshold `self.p`.
      k = int((batch_next_tkids_sort_pd.cumsum(dim=1) <= self.p).sum().item())

      # Sometimes the highest probability is larger than `self.p`, which means model is highly confident on predicting
      # next token id.  Thus the above calculation will result in `k == 0`.  In that case we only choose the token id
      # with the highest probability, we do this by setting `k = 1`.
      if k == 0:
        k = 1

      # The previous `k` token ids in `batch_next_tkids_sort` have cumulative probability less than or equal to
      # `self.p`.  We fetch them and perform further sampling.
      # shape: (1, k)
      batch_next_tkids_sort_pd = batch_next_tkids_sort_pd[:, :k]
      batch_next_tkids_sort = batch_next_tkids_sort[:, :k]

      # Use the top-k highest probabilities to construct multinomial distribution.  Then sample token id from
      # multinomial distribution as the next token id prediction result.
      # `batch_next_tkids_topk_sample` shape: (1, 1).
      batch_next_tkids_topk_sample = torch.multinomial(batch_next_tkids_sort_pd, num_samples=1)

      # Use sampled result to fetch next token id prediction.
      # shape: (1).
      batch_next_tkids = torch.gather(
        input=batch_next_tkids_sort,
        dim=1,
        index=batch_next_tkids_topk_sample,
      ).squeeze(1)
      gen_tkid = int(batch_next_tkids.item())
      gen_tkids.append(gen_tkid)

      # Update input token ids.
      batch_cur_tkids = batch_next_tkids

      # If the prediction token id is `[eos]`, then stop generation immediately.
      if gen_tkid == tknzr.eos_tkid:
        break

    # Output generated text.
    return tknzr.batch_dec(batch_tkids=[gen_tkids], rm_sp_tks=True)[0]

  @classmethod
  def infer_parser(cls, parser: argparse.ArgumentParser) -> None:
    """CLI arguments parser for language model text generation with top-k inference method.

    Parameters
    ----------
    parser: argparse.ArgumentParser
      CLI arguments parser.

    Returns
    -------
    None

    See Also
    --------
    lmp.script.gen_txt
      Use pre-trained language model checkpoint to generate continual text of given text segment.

    Examples
    --------
    >>> import argparse
    >>> from lmp.infer import TopPInfer
    >>> parser = argparse.ArgumentParser()
    >>> TopPInfer.infer_parser(parser)
    >>> args = parser.parse_args([
    ...   '--ckpt', '5000',
    ...   '--exp_name', 'my_exp',
    ...   '--max_seq_len', '128',
    ...   '--p', '0.9',
    ...   '--txt', 'Hello world',
    ... ])
    >>> args.ckpt == 5000
    True
    >>> args.exp_name == 'my_exp'
    True
    >>> args.max_seq_len == 128
    True
    >>> args.p == 0.9
    True
    >>> args.txt == 'Hello world'
    True
    >>> args.seed == 42
    True
    """
    # Load common arguments.
    super().infer_parser(parser=parser)

    # Required arguments.
    group = parser.add_argument_group('top-p inference method arguments')
    group.add_argument(
      '--p',
      help='Cumulative probability threshold.',
      required=True,
      type=float,
    )
