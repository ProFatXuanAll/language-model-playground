r"""Use pre-trained language model to calculate perplexity on given text.

One must first run the script :py:mod:`lmp.script.train_model` before running this script.

See Also
--------
:doc:`lmp.model </model/index>`
  All available language models.
:doc:`lmp.script.eval_dset_ppl </script/eval_dset_ppl>`
  Use pre-trained language model to calculate average perplexity on a particular dataset.
:doc:`lmp.script.train_model </script/train_model>`
  Train language model.

Examples
--------
The following example used pre-trained language model under experiment ``my_model_exp`` to calculate perplexity of
given text ``"Hello world"``.
It use checkpoint number ``5000`` to perform evaluation.

.. code-block::

  python -m lmp.script.eval_txt_ppl \
    --ckpt 5000 \
    --exp_name my_model_exp \
    --txt "Hello world"

The following example calculate perplexity using the last checkpoint.

.. code-block::

  python -m lmp.script.eval_txt_ppl \
    --ckpt -1 \
    --exp_name my_model_exp \
    --txt "Hello world"

You can use ``-h`` or ``--help`` options to get a list of supported CLI arguments.

.. code-block:: shell

  python -m lmp.script.eval_txt_ppl -h
"""

import argparse
import gc
import sys
from typing import List

import torch

import lmp.model
import lmp.util.cfg
import lmp.util.metric
import lmp.util.model
import lmp.util.rand
import lmp.util.tknzr
import lmp.util.validate
from lmp.tknzr._base import PAD_TKID


def parse_args(argv: List[str]) -> argparse.Namespace:
  """Parse CLI arguments.

  Parameters
  ----------
  argv: list[str]
    List of CLI arguments.

  See Also
  --------
  sys.argv
    Python CLI arguments interface.

  Returns
  -------
  argparse.Namespace
    Parsed CLI arguments.
  """
  # Create parser.
  parser = argparse.ArgumentParser(
    'python -m lmp.script.eval_txt_ppl',
    description='Use pre-trained language model to calculate perplexity on given text.',
  )

  # Required arguments.
  parser.add_argument(
    '--ckpt',
    help='Pre-trained language model checkpoint.',
    required=True,
    type=int,
  )
  parser.add_argument(
    '--exp_name',
    help='Pre-trained language model experiment name.',
    required=True,
    type=str,
  )
  parser.add_argument(
    '--txt',
    help='Text to calculate perplexity.',
    required=True,
    type=str,
  )

  # Optional arguments.
  parser.add_argument(
    '--seed',
    default=42,
    help='Random seed.  Default is ``42``.',
    type=int,
  )

  return parser.parse_args(argv)


def main(argv: List[str]) -> None:
  """Script entry point.

  Parameters
  ----------
  argv: list[str]
    List of CLI arguments.

  Returns
  -------
  None
  """
  # Parse CLI arguments.
  args = parse_args(argv=argv)

  # `args.ckpt` validation.
  lmp.util.validate.raise_if_wrong_ordered(vals=[-1, args.ckpt], val_names=['-1', 'args.ckpt'])
  # `args.txt` validation.
  lmp.util.validate.raise_if_empty_str(val=args.txt, val_name='args.txt')

  # Set random seed for reproducibility.
  lmp.util.rand.set_seed(seed=args.seed)

  # Get model running device.
  device = torch.device('cpu')
  if torch.cuda.is_available():
    device = torch.device('cuda')

  # Load pre-trained model configuration.
  model_cfg = lmp.util.cfg.load(exp_name=args.exp_name)

  # Load pre-trained tokenizer instance.
  tknzr = lmp.util.tknzr.load(exp_name=model_cfg.tknzr_exp_name)

  # Load pre-trained model instance.
  model = lmp.util.model.load(ckpt=args.ckpt, exp_name=args.exp_name)

  # Set model to evaluation mode.
  # This turn off dropout layers in model.
  model = model.eval()

  # Move model to running device.
  model = model.to(device)

  # Encode text into token ids.
  # We convert token ids into tensor and move to the same running device as model.
  batch_tkids = torch.LongTensor(tknzr.batch_enc(batch_txt=[args.txt], max_seq_len=model_cfg.max_seq_len)).to(device)

  # Loop through mini-batch by context windows.
  batch_prev_states = None
  batch_tkids_pd_list = []
  batch_next_tkids_list = []
  for ctx_idx in range(0, model_cfg.max_seq_len, model_cfg.ctx_win):
    # Fetch context window.
    ctx_batch_tkids = batch_tkids[..., ctx_idx:ctx_idx + model_cfg.ctx_win + 1]

    # Drop the remaining sequence-length-1 context window.
    if ctx_batch_tkids.size(1) == 1:
      break

    # Skip all-paddings batch.
    if torch.all(ctx_batch_tkids == PAD_TKID):
      break

    # Construct language model evaluation format.
    batch_cur_tkids = ctx_batch_tkids[..., :-1]
    batch_next_tkids = ctx_batch_tkids[..., 1:]

    # Get next token id probability distribution.
    batch_tkids_pd, batch_prev_states = model.pred(
      batch_cur_tkids=batch_cur_tkids,
      batch_prev_states=None,
    )

    batch_tkids_pd_list.append(batch_tkids_pd)
    batch_next_tkids_list.append(batch_next_tkids)

  # Calculate perplexity.
  ppl = lmp.util.metric.ppl(
    batch_tkids=torch.cat(batch_next_tkids_list, dim=1),
    batch_tkids_pd=torch.cat(batch_tkids_pd_list, dim=1),
  )

  # Output perplexity on given sample.
  print(ppl.item())

  # Free memory.
  # This is only need for unit test.
  del args
  del batch_cur_tkids
  del batch_next_tkids
  del batch_next_tkids_list
  del batch_prev_states
  del batch_tkids
  del batch_tkids_pd
  del batch_tkids_pd_list
  del ctx_batch_tkids
  del ctx_idx
  del device
  del model
  del model_cfg
  del ppl
  del tknzr
  torch.cuda.empty_cache()
  gc.collect()


if __name__ == '__main__':
  main(argv=sys.argv[1:])
