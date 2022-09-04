"""Top-1 inference method."""

from typing import ClassVar, List

import torch

from lmp.infer._base import BaseInfer
from lmp.model import BaseModel
from lmp.tknzr._base import EOS_TKID, PAD_TKID, BaseTknzr


class Top1Infer(BaseInfer):
  """Top-1 inference method.

  For each inference step, this method pick the token id with **maximum (top-1) probability** from next token id
  probability distribution over tokenizer's vocabulary as the next token id prediction.
  If there are multiple token ids having the same maximum probability, then this method pick the **smallest** token id.
  It is a greedy algorithm, simple but lack of diversity.

  Attributes
  ----------
  infer_name: ClassVar[str]
    CLI name of top-1 inference method is ``top-1``.

  See Also
  --------
  :doc:`lmp.infer </infer/index>`
    All available inference methods.
  :doc:`lmp.script.gen_txt </script/gen_txt>`
    Use pre-trained language model checkpoint to generate continual text of given text segment.
  """

  infer_name: ClassVar[str] = 'top-1'

  @torch.no_grad()
  def gen(self, model: BaseModel, tknzr: BaseTknzr, txt: str) -> str:
    """Generate continual text conditioned on given text segment.

    Top-1 inference algorithm is structured as follow:

    #. Encode input text as 1 sequence batch.
    #. Remove token ids after ``<eos>`` since model is not trained to predict tokens after seeing ``<eos>``.
    #. Loop over conditional token ids to generate conditional hidden states.
    #. Loop to generate token ids.
       In each iteration, generated token id was choosed so that it has maximum probability from next token id
       probability distribution.
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

    See Also
    --------
    :doc:`lmp.script.gen_txt </script/gen_txt>`
      Use pre-trained language model checkpoint to generate continual text of given text segment.
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
      # Get next token id probability distribution.
      # shape: (1, 1, V).
      batch_next_tkids_pd, batch_cur_states = model.pred(
        batch_cur_tkids=batch_cur_tkids,
        batch_prev_states=batch_prev_states,
      )

      # Fetch the token id with maximum probability from next token id probability distribution.
      # shape: (1, 1).
      batch_next_tkids = batch_next_tkids_pd.argmax(dim=2, keepdim=True).squeeze(1)
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
