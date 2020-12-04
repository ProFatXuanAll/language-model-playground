r"""Top 1 inference method."""

from typing import ClassVar

import torch

from lmp.infer._base import BaseInfer
from lmp.model import BaseModel
from lmp.tknzr import BaseTknzr


class Top1Infer(BaseInfer):
    r"""Top 1 inference method.

    Use index with maximum probability as next token id prediction.
    It is a greedy algorithm, simple but lack of dynamic.

    Attributes
    ==========
    infer_name: ClassVar[str]
        Inference method name is ``top-1``.
        Used for command line argument parsing.
    """
    infer_name: ClassVar[str] = 'top-1'

    @torch.no_grad()
    def gen(
            self,
            model: BaseModel,
            tknzr: BaseTknzr,
            txt: str,
    ) -> str:
        r"""Generate text conditional on text segment.

        Top 1 inference algorithm is structured as follow:

        #. Encode input text as 1 sample batch.
           (shape: ``(1, S')``)
        #. Remove ``[eos]`` token since model is not trained to predict tokens
           after seeing ``[eos]``.
           (shape: ``(1, S'-1)`` or ``(1, S)`` where ``S'-1 = S``)
        #. Truncate text to satisfy maximum sequence constraint.
           (shape: ``(1, S)`` or ``(1, max_seq_len)``)
        #. Use for-loop to generate sequence of token ids.

            #. Use ``model.pred()`` to get next token ids probability
               distribution.
               (shape: ``(1, S, V)``)
            #. Get the last next token id probability distribution.
               (shape: ``(1, V)``)
            #. Get the index with maximum probability as the last next token id
               prediction result.
               (shape: ``(1, 1)``)
            #. Concate the last next token id prediction result with previous
               next token id prediction result.
               (shape: ``(1, S+1)``)
            #. Break loop if token ids sequence length violate
               ``self.max_seq_len`` constraint.
            #. Break loop if the last next token id prediction is ``[eos]``.
            #. Otherwise go to the for-loop start and continue generation.

        #. Decode generated sequence of token ids to text and return.

        Parameters
        ==========
        model: lmp.model.BaseModel
            Pre-trained language model to generate text.
        tknzr: lmp.tknzr.BaseTknzr
            Pre-trained tokenizer for text segment encoding.
        txt: str
            Text segment to condition on.

        Returns
        =======
        str
            Generated text.
        """
        # Encode as 1 sample batch.
        batch_prev_tkids = tknzr.batch_enc(batch_txt=[txt], max_seq_len=-1)

        # Convert to tensor with `dtype == torch.int64`.
        # Tensor shape: `(1, S')`.
        # Tensor dtype: `torch.int64`.
        batch_prev_tkids = torch.LongTensor(batch_prev_tkids)

        # Remove `[eos]` token id since model is not trained to predict tokens
        # after seeing `[eos]`.
        # Tensor shape: `(1, S'-1)` or `(1, S)`.
        # Tensor dtype: `torch.int64`.
        batch_prev_tkids = batch_prev_tkids[..., :-1]

        # Satisty maximum sequence length constraint.
        # If sequence length is longer than constraint, then truncate tensor
        # to have shape `(1, self.max_seq_len)`.
        # Otherwise tensor shape remain the same.
        batch_prev_tkids = batch_prev_tkids[..., :self.max_seq_len]

        # Get model running device.
        device = next(model.parameters()).device

        # Move tensors to model running device.
        batch_prev_tkids = batch_prev_tkids.to(device)

        # Calculate how many token can be generate at most.
        # `out_seq_len` satisfy `0 <= out_seq_len <= self.max_seq_len`.
        out_seq_len = self.max_seq_len - batch_prev_tkids.size(1)

        # Generate tokens.
        for _ in range(out_seq_len):
            # Get probability distribution with current token ids.
            # Input tensor : Current token ids.
            # Input shape  : `(1, S)`.
            # Input dtype  : `torch.int64`.
            # Output tensor: Next token ids probability distribution.
            # Output shape : `(1, S, V)`.
            # Output dtype : `torch.float32`.
            batch_next_tkids_probs = model.pred(
                batch_prev_tkids=batch_prev_tkids
            )

            # Get the last token id probability distribution.
            # Only need the last token since we already know every previous
            # token ids.
            # Input tensor : Next token ids probability distribution.
            # Input shape  : `(1, S, V)`.
            # Input dtype  : `torch.float32`.
            # Output tensor: The last next token id probability distribution.
            # Output shape : `(1, V)`.
            # Output dtype : `torch.float32`.
            batch_next_tkid_probs = batch_next_tkids_probs[:, -1]

            # Use token id with the largest probability among the rest as
            # next token prediction result.
            # Input tensor : The last next token id probability distribution.
            # Input shape  : `(1, V)`.
            # Input dtype  : `torch.float32`.
            # Output tensor: The last next token id prediction result.
            # Output shape : `(1, 1)`.
            # Output dtype : `torch.int64`.
            batch_next_tkid = batch_next_tkid_probs.argmax(
                dim=-1,
                keepdim=True,
            )

            # Concate the last next token id prediction result with previous
            # token ids prediction result and use to perform further
            # prediction.
            # `batch_prev_tkids` shape: `(1, S)`.
            # `batch_prev_tkids` dtype: `torch.int64`.
            # `batch_next_tkid`  shape: `(1, 1)`.
            # `batch_next_tkid`  dtype: `torch.int64`.
            # Output shape            : `(1, S+1)`.
            # Output dtype            : `torch.int64`.
            batch_prev_tkids = torch.cat(
                [batch_prev_tkids, batch_next_tkid],
                dim=-1
            )

            # If the prediction token id is `[eos]`, then stop prediction.
            if batch_next_tkid[0, 0].item() == tknzr.eos_tkid:
                break

        # Output generated text.
        return tknzr.batch_dec(
            batch_tkids=batch_prev_tkids.tolist(),
            rm_sp_tks=True,
        )[0]
