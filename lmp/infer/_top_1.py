"""Top-1 inference method."""

from typing import ClassVar, List

import torch

from lmp.infer._base import BaseInfer
from lmp.model import BaseModel
from lmp.tknzr._base import EOS_TKID, PAD_TKID, BaseTknzr


class Top1Infer(BaseInfer):
  """Top-1 inference method.

  For each inference step, this method pick the token id with **maximum probability (top-1)** from token id probability
  distribution, and use that token id as the next token id prediction result.  If there are multiple token ids has the
  same maximum probability, this method pick the token id with the **smallest index** (which correspond to the token
  has higher occurrence count, see :py:meth:`lmp.tknzr.BaseTknzr.build_vocab`).  It is a greedy algorithm, very simple
  but lack of diversity.

  Attributes
  ----------
  infer_name: ClassVar[str]
    CLI name of top-1 inference method is ``top-1``.

  See Also
  --------
  :doc:`lmp.infer </infer/index>`
    All available inference methods.
  lmp.script.gen_txt
    Use pre-trained language model checkpoint to generate continual text of given text segment.
  """

  infer_name: ClassVar[str] = 'top-1'

  @torch.no_grad()
  def gen(self, model: BaseModel, tknzr: BaseTknzr, txt: str) -> str:
    """Generate continual text conditioned on given text segment.

    Top-1 inference algorithm is structured as follow:

    #. Encode input text as 1 sample batch.
    #. Remove token ids after ``[eos]`` since model is not trained to predict tokens after seeing ``[eos]``.
    #. Loop over conditioned token ids to generate conditioned hidden states.
    #. Loop to generate token ids.  In each iteration, generated token id was choosed so that it has maximum
       probability from next token id prediction probability distribution.  Generating loop will stop early if
       ``[eos]`` is generated, otherwise generating loop only stop when maximum length constraint enforced by
       ``self.max_seq_len`` is violated.
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

    See Also
    --------
    lmp.script.gen_txt
      Use pre-trained language model checkpoint to generate continual text of given text segment.
    """
    # Get model running device.
    device = next(model.parameters()).device

    # Encode as 1 sample batch.  We convert token ids to tensor and move tensor to the same running device as model.
    # shape: (1, max_seq_len).
    batch_cur_tkids = torch.LongTensor(tknzr.batch_enc(batch_txt=[txt], max_seq_len=self.max_seq_len)).to(device)

    # Remove token ids after `[eos]` since model is not trained to predict tokens after seeing `[eos]`.
    mask = (batch_cur_tkids == EOS_TKID) | (batch_cur_tkids == PAD_TKID)
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

      # Fetch the token id with maximum probability from next token id prediction probability distribution.
      # shape: (1).
      batch_next_tkids = batch_next_tkids_pd.argmax(dim=1, keepdim=True).squeeze(1)
      gen_tkid = int(batch_next_tkids.item())
      gen_tkids.append(gen_tkid)

      # Update input token ids.
      batch_cur_tkids = batch_next_tkids

      # If the prediction token id is `[eos]`, then stop generation immediately.
      if gen_tkid == EOS_TKID:
        break

    # Output generated text.
    return tknzr.batch_dec(batch_tkids=[gen_tkids], rm_sp_tks=True)[0]
