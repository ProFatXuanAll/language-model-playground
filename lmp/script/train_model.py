r"""Use this script to train language model on particular dataset.

This script is usually run after training tokenizer.
Training performance will be shown on both CLI and tensorboard.
Use ``pipenv run tensorboard`` to launch tensorboard and open browser with URL http://localhost:6006/ to see model
training performance.

See Also
--------
:doc:`lmp.model </model/index>`
  All available language models.
:doc:`lmp.script.train_tknzr </script/train_tknzr>`
  Tokenizer training script.

Examples
--------
The following example script train Elman Net model :py:class:`lmp.model.ElmanNet` on Wiki-Text-2 dataset
:py:class:`lmp.dset.WikiText2Dset` with ``train`` version.

.. code-block:: shell

  python -m lmp.script.train_model Elman-Net \
    --batch_size 32 \
    --beta1 0.9 \
    --beta2 0.99 \
    --ckpt_step 1000 \
    --ctx_win 32 \
    --d_emb 100 \
    --d_hid 100 \
    --dset_name wiki-text-2 \
    --eps 1e-8 \
    --exp_name my_model_exp \
    --log_step 200 \
    --lr 1e-4 \
    --max_norm 1 \
    --max_seq_len 128 \
    --n_lyr 1 \
    --p_emb 0.5 \
    --p_hid 0.1 \
    --tknzr_exp_name my_tknzr_exp \
    --total_step 50000 \
    --ver train \
    --warmup_step 10000 \
    --wd 1e-2

The training result will be save at path ``project_root/exp/my_model_exp`` and can be reused by other scripts.
We only save checkpoints per ``--ckpt_step`` steps and log performance per ``--log_step`` steps.

One can increase ``--total_step`` to train more steps.
Be careful model might overfit on datasets if model were trained with too many steps.

.. code-block:: shell

  python -m lmp.script.train_model Elman-Net \
    --batch_size 32 \
    --beta1 0.9 \
    --beta2 0.99 \
    --ckpt_step 1000 \
    --ctx_win 32 \
    --d_emb 100 \
    --d_hid 100 \
    --dset_name wiki-text-2 \
    --eps 1e-8 \
    --exp_name my_model_exp \
    --log_step 200 \
    --lr 1e-4 \
    --max_norm 1 \
    --max_seq_len 128 \
    --n_lyr 1 \
    --p_emb 0.5 \
    --p_hid 0.1 \
    --tknzr_exp_name my_tknzr_exp \
    --total_step 100000 \
    --ver train \
    --warmup_step 10000 \
    --wd 1e-2

One can reduce overfitting with the following ways:

- Increase ``--batch_size``.  This increase sample variance and make model hard to optimize.
- Increase ``--wd``.  This increase L2 penalty and make model output differences small when given large variance input.
- Reduce model parameters (In :py:class:`lmp.model.ElmanNet` this means reducing ``--d_emb`` or ``--d_hid``).
  This make model capacity low and hard to memorize all samples.
  Thus model is forced to learn and utilize patterns found on different samples.
- Use dropout (In :py:class:`lmp.model.ElmanNet` this means increasing ``--p_emb`` or ``--p_hid``).
  Dropout is a way to perform models ensembling without the cost of training multiple model instances.
- Use any combinations of tricks above.

.. code-block:: shell

  python -m lmp.script.train_model Elman-Net \
    --batch_size 32 \
    --beta1 0.9 \
    --beta2 0.99 \
    --ckpt_step 1000 \
    --ctx_win 32 \
    --d_emb 50 \
    --d_hid 50 \
    --dset_name wiki-text-2 \
    --eps 1e-8 \
    --exp_name my_model_exp \
    --log_step 200 \
    --lr 1e-4 \
    --max_norm 1 \
    --max_seq_len 128 \
    --n_lyr 1 \
    --p_emb 0.5 \
    --p_hid 0.5 \
    --tknzr_exp_name my_tknzr_exp \
    --total_step 50000 \
    --ver train \
    --warmup_step 10000 \
    --wd 1e-1

We use :py:class:`torch.optim.AdamW` to perform optimization.
Use ``--beta1``, ``--beta2``, ``--eps``, ``--lr`` and ``--wd`` to adjust optimization hyperparameters.
We also use ``--max_norm`` to perform gradient clipping which avoids gradient explosion.

.. code-block:: shell

  python -m lmp.script.train_model Elman-Net \
    --batch_size 32 \
    --beta1 0.95 \
    --beta2 0.98 \
    --ckpt_step 1000 \
    --ctx_win 32 \
    --d_emb 100 \
    --d_hid 100 \
    --dset_name wiki-text-2 \
    --eps 1e-6 \
    --exp_name my_model_exp \
    --log_step 200 \
    --lr 5e-4 \
    --max_norm 0.1 \
    --max_seq_len 128 \
    --p_emb 0.5 \
    --p_hid 0.1 \
    --tknzr_exp_name my_tknzr_exp \
    --total_step 50000 \
    --ver train \
    --warmup_step 10000 \
    --wd 1e-2

You can use ``-h`` or ``--help`` options to get a list of available language models.

.. code-block:: shell

  python -m lmp.script.train_model -h

You can use ``-h`` or ``--help`` options on a specific language model to get a list of supported CLI arguments.

.. code-block:: shell

  python -m lmp.script.train_model Elman-Net -h
"""

import argparse
import copy
import gc
import sys
from typing import List

import torch
import torch.nn.utils
import torch.optim
import torch.utils.data
# Typeshed for `tqdm` is not available, we ignore type check on `tqdm`.
from tqdm import tqdm  # type: ignore

import lmp.dset
import lmp.model
import lmp.util.cfg
import lmp.util.dset
import lmp.util.log
import lmp.util.model
import lmp.util.optim
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
  parser = argparse.ArgumentParser('python -m lmp.script.train_model', description='Train language model.')

  # Use model name to create subparser for all language models.
  subparsers = parser.add_subparsers(dest='model_name', required=True)
  for model_name, model_type in lmp.model.MODEL_OPTS.items():
    model_subparser = subparsers.add_parser(
      model_name,
      description=f'Training `lmp.model.{model_type.__name__}` language model.',
    )

    # Required arguments.
    group = model_subparser.add_argument_group('language model training arguments')
    group.add_argument(
      '--batch_size',
      help='Mini-batch size.',
      required=True,
      type=int,
    )
    group.add_argument(
      '--beta1',
      help='First beta coefficient of AdamW optimizer.',
      required=True,
      type=float,
    )
    group.add_argument(
      '--beta2',
      help='Second beta coefficient of AdamW optimizer.',
      required=True,
      type=float,
    )
    group.add_argument(
      '--ckpt_step',
      help='Checkpoint save interval.',
      required=True,
      type=int,
    )
    group.add_argument(
      '--ctx_win',
      help='Context window size.',
      required=True,
      type=int,
    )
    group.add_argument(
      '--dset_name',
      choices=lmp.dset.DSET_OPTS.keys(),
      help='Name of the dataset which will be used to train language model.',
      required=True,
      type=str,
    )
    group.add_argument(
      '--eps',
      help='Denominator smooth term of AdamW optimizer.',
      required=True,
      type=float,
    )
    group.add_argument(
      '--exp_name',
      help='Name of the language model training experiment.',
      required=True,
      type=str,
    )
    group.add_argument(
      '--log_step',
      help='Performance log interval.',
      required=True,
      type=int,
    )
    group.add_argument(
      '--lr',
      help='Learning rate.',
      required=True,
      type=float,
    )
    group.add_argument(
      '--max_norm',
      help='Gradient max-norm constraint.',
      required=True,
      type=float,
    )
    group.add_argument(
      '--max_seq_len',
      help='Maximum sequence length constraint.',
      required=True,
      type=int,
    )
    group.add_argument(
      '--tknzr_exp_name',
      help='Name of the pre-trained tokenizer experiment.',
      required=True,
      type=str,
    )
    group.add_argument(
      '--total_step',
      help='Number of training steps.',
      required=True,
      type=int,
    )
    group.add_argument(
      '--ver',
      help='Version of the dataset.',
      required=True,
      type=str,
    )
    group.add_argument(
      '--warmup_step',
      help='Learning rate warm up steps.',
      required=True,
      type=int,
    )
    group.add_argument(
      '--wd',
      help='Weight decay coefficient of AdamW optimizer.',
      required=True,
      type=float,
    )

    # Optional arguments.
    group.add_argument(
      '--seed',
      default=42,
      help='Random seed.  Default is ``42``.',
      type=int,
    )

    # Add model specific arguments.
    model_type.add_CLI_args(parser=model_subparser)

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

  # `args.batch_size` validation.
  lmp.util.validate.raise_if_wrong_ordered(vals=[1, args.batch_size], val_names=['1', 'args.batch_size'])
  # `args.ckpt_step` validation.
  lmp.util.validate.raise_if_wrong_ordered(vals=[1, args.ckpt_step], val_names=['1', 'args.ckpt_step'])
  # `args.ctx_win` validation.
  lmp.util.validate.raise_if_wrong_ordered(
    vals=[1, args.ctx_win, args.max_seq_len],
    val_names=['1', 'args.ctx_win', 'args.max_seq_len'],
  )
  # `args.log_step` validation.
  lmp.util.validate.raise_if_wrong_ordered(vals=[1, args.log_step], val_names=['1', 'args.log_step'])
  # `args.max_norm` validation.
  lmp.util.validate.raise_if_wrong_ordered(vals=[0, args.max_norm], val_names=['0', 'args.max_norm'])

  # Save training configuration.
  lmp.util.cfg.save(args=args, exp_name=args.exp_name)

  # Set random seed for reproducibility.
  lmp.util.rand.set_seed(seed=args.seed)

  # Get model running device.
  device = torch.device('cpu')
  if torch.cuda.is_available():
    device = torch.device('cuda')

  # Load pre-trained tokenizer.
  tknzr = lmp.util.tknzr.load(exp_name=args.tknzr_exp_name)

  # Get dataset instance of specific version.
  dset = lmp.util.dset.load(**args.__dict__)

  # Mini-batch random sampler.
  data_loader = torch.utils.data.DataLoader(
    batch_size=args.batch_size,
    dataset=dset,
    shuffle=True,
  )

  # Create new model instance and initialize model parameters.
  model = lmp.util.model.create(tknzr=tknzr, **args.__dict__)
  model = model.train()
  model.params_init()

  # Move model to running device.
  model = model.to(device)

  # Get new optimizer instance.
  optim = lmp.util.optim.get_optimizer(
    beta1=args.beta1,
    beta2=args.beta2,
    eps=args.eps,
    lr=args.lr,
    model=model,
    wd=args.wd,
  )

  # Get learning rate scheduler.
  schdl = lmp.util.optim.get_scheduler(
    optim=optim,
    total_step=args.total_step,
    warmup_step=args.warmup_step,
  )

  # Get tensorboard logger instance.
  writer = lmp.util.log.get_tb_logger(exp_name=args.exp_name)

  # Log performance target.
  avg_loss = 0.0

  # Get CLI logger instance.
  cli_logger = tqdm(range(args.total_step), desc=f'loss: {avg_loss:.6f}', dynamic_ncols=True)

  # Global optimization step.
  step = 0
  while step < args.total_step:
    # Loop through dataset by mini-batches.
    for batch_txt in data_loader:
      # Encode batch text into batch token ids.
      # We convert batch token ids into tensor and move to tensor to the same running device as model.
      batch_tkids = torch.LongTensor(tknzr.batch_enc(batch_txt=batch_txt, max_seq_len=args.max_seq_len)).to(device)

      # Loop through mini-batch by context windows.
      batch_prev_states = None
      for ctx_idx in range(0, args.max_seq_len, args.ctx_win):
        # Fetch context window.
        ctx_batch_tkids = batch_tkids[..., ctx_idx:ctx_idx + args.ctx_win + 1]

        # Drop the remaining sequence-length-1 context window.
        if ctx_batch_tkids.size(1) == 1:
          break

        # Skip all-paddings batches.
        if torch.all(ctx_batch_tkids == PAD_TKID):
          break

        # Construct language model training format.
        batch_cur_tkids = ctx_batch_tkids[..., :-1]
        batch_next_tkids = ctx_batch_tkids[..., 1:]

        # Calculate next token prediction loss.
        loss, batch_prev_states = model.loss(
          batch_cur_tkids=batch_cur_tkids,
          batch_next_tkids=batch_next_tkids,
          batch_prev_states=batch_prev_states,
        )

        # Accumulate average loss for logging.
        # Use `.item()` to avoid construct tensor graph.
        avg_loss += loss.item()

        # Perform backward pass / back propagation.
        loss.backward()

        # Perform gradient clipping to avoid gradient explosion.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)

        # Gradient descent.
        optim.step()

        # Update learning rate.
        schdl.step()

        # Clean up gradient.
        optim.zero_grad()

        # Increment global step.
        step += 1

        # Save checkpoint for each `ckpt_step` step.
        # We move model to CPU first then move back to CUDA device.
        if step % args.ckpt_step == 0:
          lmp.util.model.save(ckpt=step, exp_name=args.exp_name, model=copy.deepcopy(model).to('cpu'))

        # Log performance for each `log_step` step.
        if step % args.log_step == 0:
          avg_loss = avg_loss / args.log_step

          # Log on CLI.
          cli_logger.set_description(f'loss: {avg_loss:.6f}')
          cli_logger.update(args.log_step)

          # Log on tensorboard.
          writer.add_scalar(f'train-loss/{args.dset_name}/{args.ver}', avg_loss, step)
          writer.add_scalar('lr', schdl.get_last_lr()[0], step)

          # Refresh log performance.
          avg_loss = 0.0

        # Only train certain number of steps.
        if step >= args.total_step:
          break

      # Only train certain number of steps.
      if step >= args.total_step:
        break

  # Save last checkpoint.
  lmp.util.model.save(ckpt=step, exp_name=args.exp_name, model=copy.deepcopy(model).to('cpu'))

  # Close tensorboard logger.
  writer.close()

  # Close CLI logger.
  cli_logger.close()

  # Free memory.
  # This is only need for unit test.
  del args
  del avg_loss
  del batch_cur_tkids
  del batch_next_tkids
  del batch_prev_states
  del batch_tkids
  del ctx_batch_tkids
  del ctx_idx
  del data_loader
  del device
  del dset
  del loss
  del model
  del optim
  del schdl
  del step
  del tknzr
  del writer
  torch.cuda.empty_cache()
  gc.collect()


if __name__ == '__main__':
  main(argv=sys.argv[1:])
