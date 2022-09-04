r"""Use pre-trained language model to calculate perplexity on given text.

One must first run the script :doc:`lmp.script.train_model </script/train_model>` before running this script.

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
The following example used pre-trained language model under experiment ``my_model_exp`` to calculate perplexity of the
given text ``"Hello world"``.
It use checkpoint number ``5000`` to perform evaluation.

.. code-block::

  python -m lmp.script.eval_txt_ppl \
    --ckpt 5000 \
    --exp_name my_model_exp \
    --txt "Hello world"

The following example calculate perplexity using the last checkpoint of experiment ``my_model_exp``.

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

  parser.add_argument(
    '--ckpt',
    default=-1,
    help='''
    Pre-trained language model checkpoint.
    Set to ``-1`` to use the last checkpoint.
    Default is ``-1``.
    ''',
    type=int,
  )
  parser.add_argument(
    '--exp_name',
    default='my_model_exp',
    help='''
    Pre-trained language model experiment name.
    Default is ``my_model_exp``.
    ''',
    type=str,
  )
  parser.add_argument(
    '--txt',
    default='hello world',
    help='''
    Text to calculate perplexity.
    Default is ``hello world``.
    ''',
    type=str,
  )
  parser.add_argument(
    '--seed',
    default=42,
    help='''
    Random seed.
    Default is ``42``.
    ''',
    type=int,
  )

  args = parser.parse_args(argv)

  # `args.ckpt` validation.
  lmp.util.validate.raise_if_wrong_ordered(vals=[-1, args.ckpt], val_names=['-1', 'args.ckpt'])
  # `args.txt` validation.
  lmp.util.validate.raise_if_empty_str(val=args.txt, val_name='args.txt')

  return args


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
  # Shape: (1, S)
  batch_tkids = torch.LongTensor([tknzr.enc(txt=args.txt)]).to(device)
  S = batch_tkids.size(1)

  # Record BPC and loop through mini-batch by context windows.
  # In practice we can use word or subword as token, so we shall call it bit-per-token instead of BPC.
  # Naming it as BPC is simply because the convention.
  batch_prev_states = None
  bpc = 0.0
  for ctx_idx in range(0, model_cfg.max_seq_len, model_cfg.max_seq_len):
    # Fetch context window.
    ctx_batch_tkids = batch_tkids[..., ctx_idx:ctx_idx + model_cfg.max_seq_len + 1]

    # Drop the remaining sequence-length-1 context window.
    if ctx_batch_tkids.size(1) == 1:
      break

    # Construct language model evaluation format.
    batch_cur_tkids = ctx_batch_tkids[..., :-1]
    batch_next_tkids = ctx_batch_tkids[..., 1:]

    # Get next token id probability distribution.
    batch_tkids_pd, batch_cur_states = model.pred(
      batch_cur_tkids=batch_cur_tkids,
      batch_prev_states=batch_prev_states,
    )

    # Calculate -p log p.
    nplogp = lmp.util.metric.nplogp(batch_tkids=batch_next_tkids, batch_tkids_pd=batch_tkids_pd, use_log2=True)

    # Record BPC.
    bpc += (nplogp / S).sum().item()

    # Update hidden states.
    batch_prev_states = batch_cur_states

  # Convert BPC to perplexity.
  ppl = pow(2, bpc)

  # Output perplexity on given sample.
  print(ppl)


if __name__ == '__main__':
  main(argv=sys.argv[1:])
