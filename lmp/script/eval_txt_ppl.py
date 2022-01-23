r"""Use pre-trained language model to calculate perplexity on given text.

One must first run the script :py:mod:`lmp.script.train_model` before running this script.

See Also
--------
lmp.model
  All available language models.
lmp.script.eval_dset_ppl
  Use pre-trained language model to calculate average perplexity on a particular dataset.
lmp.script.train_model
  Train language model.

Examples
--------
The following example used pre-trained language model under experiment ``my_model_exp`` to calculate perplexity of
given text ``"Hello world"``.  It use checkpoint number ``5000`` to perform evaluation.

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
import sys
from typing import List

import torch

import lmp.model
import lmp.util.cfg
import lmp.util.metric
import lmp.util.model
import lmp.util.rand
import lmp.util.tknzr


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
    help='Random seed.',
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

  # Set random seed for reproducibility.
  lmp.util.rand.set_seed(seed=args.seed)

  # Load pre-trained model configuration.
  model_cfg = lmp.util.cfg.load(exp_name=args.exp_name)

  # Load pre-trained tokenizer configuration.
  tknzr_cfg = lmp.util.cfg.load(exp_name=model_cfg.tknzr_exp_name)

  # Load pre-trained tokenizer instance.
  tknzr = lmp.util.tknzr.load(exp_name=tknzr_cfg.exp_name, tknzr_name=tknzr_cfg.tknzr_name)

  # Load pre-trained model instance.
  model = lmp.util.model.load(ckpt=args.ckpt, exp_name=args.exp_name, tknzr=tknzr)

  # Set model to evaluation model.  This turn off dropout layers in model.
  model = model.eval()

  # Get model running device.
  device = torch.device('cpu')
  if torch.cuda.is_available():
    device = torch.device('cuda')

  # Move model to running device.
  model = model.to(device)

  # Encode text into token ids.  We convert token ids into tensor and move to the same running device as model.
  batch_tkids = torch.LongTensor(tknzr.batch_enc(batch_txt=[args.txt], max_seq_len=model_cfg.max_seq_len)).to(device)

  # Format token ids to satisfy language model training format.
  batch_cur_tkids = batch_tkids[..., :-1]
  batch_next_tkids = batch_tkids[..., 1:]

  # Loop over token ids to get next token id prediction probability distribution.
  batch_prev_states = None
  batch_tkids_pd = []
  for i in range(batch_cur_tkids.size(1)):
    batch_next_tkids_pd, batch_prev_states = model.pred(
      batch_cur_tkids=batch_cur_tkids[:, i],
      batch_prev_states=batch_prev_states,
    )

    # Collect prediction probability distribution.
    batch_tkids_pd.append(batch_next_tkids_pd)

  # Calculate perplexity.
  ppl = lmp.util.metric.ppl(
    batch_tkids=batch_next_tkids,
    batch_tkids_pd=torch.stack(batch_tkids_pd, dim=1),
    eos_tkid=tknzr.eos_tkid,
    pad_tkid=tknzr.pad_tkid,
  )

  # Output perplexity on given sample.
  print(ppl.item())


if __name__ == '__main__':
  main(argv=sys.argv[1:])
