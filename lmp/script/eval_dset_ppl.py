r"""Use pre-trained language model checkpoints to calculate perplexity on a dataset.

One must first run the script :doc:`lmp.script.train_model </script/train_model>` before running this script.
Use ``pipenv run tensorboard`` to launch tensorboard and open browser with URL http://localhost:6006/ to see evaluation
results over selected model checkpoints.

See Also
--------
:doc:`lmp.model </model/index>`
  All available language models.
:doc:`lmp.script.eval_txt_ppl </script/eval_txt_ppl>`
  Use pre-trained language model to calculate perplexity on given text.
:doc:`lmp.script.train_model </script/train_model>`
  Train language model.

Examples
--------
The following example evaluate language model experiment ``my_model_exp`` on :py:class:`~lmp.dset.WikiText2` dataset
with version ``valid``.
It evaluates checkpoints whose numbers are larger than or equal to ``5000``.

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

There are many checkpoints to be evaluated.
One can specify checkpoint range one want to evaluate.

.. code-block::

  python -m lmp.script.eval_dset_ppl wiki-text-2 \
    --batch_size 32 \
    --first_ckpt 5000 \
    --exp_name my_model_exp \
    --last_ckpt 10000 \
    --ver valid

Since evaluation do not need to construct tensor graph when perform forward pass, model will consume less memory than
training.
Thus we can use larger batch size to accelerate evaluation process.

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

    dset_subparser.add_argument(
      '--batch_size',
      default=32,
      help='''
      Evaluation mini-batch size.
      Default is ``32``.
      ''',
      type=int,
    )
    dset_subparser.add_argument(
      '--exp_name',
      default='my_model_exp',
      help='''
      Pre-trained language model experiment name.
      Default is ``my_model_exp``.
      ''',
      type=str,
    )
    dset_subparser.add_argument(
      '--first_ckpt',
      default=0,
      help='''
      The first checkpoint of pre-trained language model to be evaluated.
      Default is ``0``.
      ''',
      type=int,
    )
    dset_subparser.add_argument(
      '--last_ckpt',
      default=-1,
      help='''
      The last checkpoint of pre-trained language model to be evaluated.
      Set to ``-1`` to include the last checkpoint.
      Default is ``-1``.
      ''',
      type=int,
    )
    dset_subparser.add_argument(
      '--seed',
      default=42,
      help='''
      Random seed.
      Default is ``42``.
      ''',
      type=int,
    )
    dset_subparser.add_argument(
      '--stride',
      default=-1,
      help='''
      Number of overlapping tokens between context windows.
      Set to ``-1`` to use model training configuration.
      Default is ``-1``.
      ''',
      type=int,
    )
    dset_subparser.add_argument(
      '--ver',
      default=None,
      help=f'''
      Version of the {dset_type.__name__} dataset.
      Default is ``{dset_type.df_ver}``.
      ''',
      choices=dset_type.vers,
      type=str,
    )

  args = parser.parse_args(argv)

  # `args.batch_size` validation.
  lmp.util.validate.raise_if_wrong_ordered(vals=[1, args.batch_size], val_names=['1', 'args.batch_size'])
  # `args.first_ckpt` validation.
  lmp.util.validate.raise_if_wrong_ordered(vals=[-1, args.first_ckpt], val_names=['-1', 'args.first_ckpt'])
  # `args.last_ckpt` validation.
  lmp.util.validate.raise_if_wrong_ordered(vals=[-1, args.last_ckpt], val_names=['-1', 'args.last_ckpt'])

  if args.ver is None:
    args.ver = lmp.util.dset.DSET_OPTS[args.dset_name].df_ver

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

  # Use model configuration.
  if args.stride == -1:
    args.stride = model_cfg.stride
  else:
    lmp.util.validate.raise_if_wrong_ordered(
      vals=[1, args.stride, model_cfg.max_seq_len],
      val_names=['1', 'args.stride', 'model_cfg.max_seq_len'],
    )

  # Load pre-trained tokenizer instance.
  tknzr = lmp.util.tknzr.load(exp_name=model_cfg.tknzr_exp_name)

  # Get dataset instance of specific version.
  dset = lmp.util.dset.load(**args.__dict__)

  # Convert dataset to language model training format.
  # Stride is set to `max_seq_len` so that no overlapping happens.
  lm_format_dset = lmp.util.dset.LMFormatDset(
    dset=dset,
    max_seq_len=model_cfg.max_seq_len,
    stride=args.stride,
    tknzr=tknzr,
  )

  # Mini-batch sampler.
  data_loader = torch.utils.data.DataLoader(
    batch_size=args.batch_size,
    dataset=lm_format_dset,
    shuffle=False,
  )

  # Get tensorboard logger instance.
  writer = lmp.util.log.get_tb_logger(exp_name=args.exp_name)

  # Evaluate checkpoints within ranges.
  best_ckpt = -1
  best_ppl = 0.0
  for ckpt in lmp.util.model.list_ckpts(exp_name=args.exp_name, first_ckpt=args.first_ckpt, last_ckpt=args.last_ckpt):
    # Load pre-trained model instance.
    model = lmp.util.model.load(ckpt=ckpt, exp_name=args.exp_name)

    # Set model to evaluation mode.
    # This turn off dropout layers in model.
    model = model.eval()

    # Move model to running device.
    model = model.to(device)

    # Record BPC (bit-per-character).
    # In practice we can use word or subword as token, so we shall call it bit-per-token instead of BPC.
    # Naming it as BPC is simply because the convention.
    bpc = 0.0
    for batch_is_not_ctx, batch_cur_tkids, batch_next_tkids in tqdm(data_loader):
      # Get next token id probability distribution.
      batch_tkids_pd, _ = model.pred(
        batch_cur_tkids=batch_cur_tkids.to(device),
        batch_prev_states=None,
      )

      # Calculate negative log-likelihood -log(p).
      nll = lmp.util.metric.nll(batch_tkids=batch_next_tkids.to(device), batch_tkids_pd=batch_tkids_pd, use_log2=True)

      # Only tokens not used as context will contribute to BPC.
      masked_nll = nll * batch_is_not_ctx.to(device)

      # Record BPC.
      bpc += (masked_nll / lm_format_dset.n_tk).sum().item()

    # Convert BPC to perplexity.
    ppl = pow(2, bpc)

    # Log dataset perplexity to CLI and tensorboard.
    writer.add_scalar(f'ppl/{args.dset_name}/{args.ver}', ppl, ckpt)
    print(f'checkpoint: {ckpt}, ppl: {ppl}')

    if best_ppl > ppl or best_ckpt == -1:
      best_ckpt = ckpt
      best_ppl = ppl

  print(f'best checkpoint: {best_ckpt}, best ppl: {best_ppl}')

  # Close tensorboard logger.
  writer.flush()
  writer.close()


if __name__ == '__main__':
  main(argv=sys.argv[1:])
