r"""Use pre-trained language model checkpoints to calculate average perplexity on a particular dataset.

One must first run the script :py:mod:`lmp.script.train_model` before running this script.  Use
``pipenv run tensorboard`` to launch tensorboard and open browser with URL http://localhost:6006/ to see evaluation
results over selected model checkpoints.

See Also
--------
lmp.model
  All available language models.
lmp.script.eval_txt_ppl
  Use pre-trained language model to calculate perplexity on given text.
lmp.script.train_model
  Train language model.

Examples
--------
The following example evaluate language model experiment ``my_model_exp`` on :py:class:`lmp.dset.WikiText2` dataset
with version ``valid``.  It evaluate on checkpoints number larger than or equal to ``5000``.

.. code-block::

   python -m lmp.script.eval_dset_ppl wiki-text-2 \
     --batch_size 32 \
     --first_ckpt 5000 \
     --exp_name my_model_exp \
     --ver valid

The following example only evaluate on the last checkpoint.

.. code-block::

   python -m lmp.script.eval_dset_ppl wiki-text-2 \
     --batch_size 32 \
     --first_ckpt -1 \
     --exp_name my_model_exp \
     --ver valid

Specify checkpoint range to be evaluated.

.. code-block::

   python -m lmp.script.eval_dset_ppl wiki-text-2 \
     --batch_size 32 \
     --first_ckpt 5000 \
     --exp_name my_model_exp \
     --last_ckpt 10000 \
     --ver valid

Since evaluation do not need to construct tensor graph when perform forward pass, model will consume less memory than
training.  Thus we can use larger batch size to accelerate evaluation process.

.. code-block::

   python -m lmp.script.eval_dset_ppl wiki-text-2 \
     --batch_size 128 \
     --ckpt -1 \
     --exp_name my_model_exp \
     --ver valid

You can use ``-h`` or ``--help`` options to get a list of supported CLI arguments.

.. code-block:: shell

   python -m lmp.script.eval_dset_ppl -h
"""

import argparse
import gc
import sys
from typing import List

import torch
import torch.utils.data
# Typeshed for `tqdm` is not available, we ignore type check on `tqdm`.
from tqdm import tqdm  # type: ignore

import lmp.dset
import lmp.model
import lmp.util.cfg
import lmp.util.dset
import lmp.util.log
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
    'python -m lmp.script.eval_dset_ppl',
    description='Use pre-trained language model checkpoints to calculate average perplexity on a particular dataset.',
  )

  # Use dataset name to create subparser for all datasets.
  subparsers = parser.add_subparsers(dest='dset_name', required=True)
  for dset_name, dset_type in lmp.dset.DSET_OPTS.items():
    dset_subparser = subparsers.add_parser(
      dset_name,
      description=f'Calculate perplexity on {dset_type.__name__} dataset.',
    )

    # Required arguments.
    dset_subparser.add_argument(
      '--batch_size',
      help='Evaluation batch size.',
      required=True,
      type=int,
    )
    dset_subparser.add_argument(
      '--exp_name',
      help='Pre-trained language model experiment name.',
      required=True,
      type=str,
    )
    dset_subparser.add_argument(
      '--first_ckpt',
      help='The first checkpoint of pre-trained language model to be evaluated.',
      required=True,
      type=int,
    )

    # Optional arguments.
    dset_subparser.add_argument(
      '--last_ckpt',
      default=-1,
      help='The last checkpoint of pre-trained language model to be evaluated.',
      type=int,
    )
    dset_subparser.add_argument(
      '--seed',
      default=42,
      help='Random seed.',
      type=int,
    )
    dset_subparser.add_argument(
      '--ver',
      default=None,
      help=f'Version of the {dset_type.__name__} dataset.  Defaults to {dset_type.df_ver}.',
      choices=dset_type.vers,
      type=str,
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

  # Get dataset instance with specified version.
  dset = lmp.util.dset.load(dset_name=args.dset_name, ver=args.ver)
  dset_size = len(dset)

  # Mini-batch random sampler.
  data_loader = torch.utils.data.DataLoader(batch_size=args.batch_size, dataset=dset, shuffle=False)

  # Load pre-trained model configuration.
  model_cfg = lmp.util.cfg.load(exp_name=args.exp_name)

  # Load pre-trained tokenizer configuration.
  tknzr_cfg = lmp.util.cfg.load(exp_name=model_cfg.tknzr_exp_name)

  # Load pre-trained tokenizer instance.
  tknzr = lmp.util.tknzr.load(exp_name=tknzr_cfg.exp_name, tknzr_name=tknzr_cfg.tknzr_name)

  # Get model running device.
  device = torch.device('cpu')
  if torch.cuda.is_available():
    device = torch.device('cuda')

  # Get tensorboard logger instance.
  writer = lmp.util.log.get_tb_logger(exp_name=args.exp_name)

  # Evaluate checkpoints within ranges.
  for ckpt in lmp.util.model.list_ckpts(exp_name=args.exp_name, first_ckpt=args.first_ckpt, last_ckpt=args.last_ckpt):
    # Load pre-trained model instance.
    model = lmp.util.model.load(ckpt=ckpt, exp_name=args.exp_name, tknzr=tknzr)

    # Set model to evaluation model.  This turn off dropout layers in model.
    model = model.eval()

    # Move model to running device.
    model = model.to(device)

    # Record average perplexity.
    avg_ppl = 0.0
    for batch_txt in tqdm(data_loader):
      # Encode text into token ids.  We convert token ids into tensor and move to the same running device as model.
      batch_tkids = torch.LongTensor(tknzr.batch_enc(batch_txt=batch_txt, max_seq_len=model_cfg.max_seq_len)).to(device)

      # Format batch token ids to satisfy language model training format.
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
      batch_ppl = lmp.util.metric.ppl(
        batch_tkids=batch_next_tkids,
        batch_tkids_pd=torch.stack(batch_tkids_pd, dim=1),
        eos_tkid=tknzr.eos_tkid,
        pad_tkid=tknzr.pad_tkid,
      )

      # Accumulate average perplexity.
      avg_ppl += (batch_ppl / dset_size).sum().item()

      # Free memory.
      del batch_cur_tkids
      del batch_next_tkids
      del batch_next_tkids_pd
      del batch_ppl
      del batch_prev_states
      del batch_tkids
      del batch_tkids_pd
      torch.cuda.empty_cache()
      gc.collect()

    # Free memory.
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # Log average perplexity on dataset to CLI and tensorboard.
    writer.add_scalar(f'ppl/{args.dset_name}/{args.ver}', avg_ppl, ckpt)
    print(f'checkpoint: {ckpt}, avg ppl: {avg_ppl}')

  # Free memory.  This is only need for unit test.
  del args
  del avg_ppl
  del ckpt
  del data_loader
  del device
  del dset
  del dset_size
  del model_cfg
  del tknzr
  del tknzr_cfg
  del writer
  gc.collect()


if __name__ == '__main__':
  main(argv=sys.argv[1:])
