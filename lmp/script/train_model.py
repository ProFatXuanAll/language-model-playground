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
The following example script train Elman Net model :py:class:`~lmp.model.ElmanNet` on Wiki-Text-2 dataset
:py:class:`~lmp.dset.WikiText2Dset` with ``train`` version.

.. code-block:: shell

  python -m lmp.script.train_model Elman-Net \
    --batch_size 32 \
    --beta1 0.9 \
    --beta2 0.999 \
    --ckpt_step 1000 \
    --d_emb 100 \
    --d_hid 100 \
    --dset_name wiki-text-2 \
    --eps 1e-8 \
    --exp_name my_model_exp \
    --init_lower -0.1 \
    --init_upper 0.1 \
    --label_smoothing 0.0 \
    --log_step 500 \
    --lr 1e-3 \
    --max_norm 10 \
    --max_seq_len 32 \
    --n_lyr 1 \
    --p_emb 0.5 \
    --p_hid 0.1 \
    --stride 32 \
    --tknzr_exp_name my_tknzr_exp \
    --total_step 10000 \
    --ver train \
    --warmup_step 5000 \
    --weight_decay 0.0

The training result will be save at path ``project_root/exp/my_model_exp`` and can be reused by other scripts.
We only save checkpoints per ``--ckpt_step`` steps and log performance per ``--log_step`` steps.

One can increase ``--total_step`` to train more steps.
Be careful model might overfit on datasets if model were trained with too many steps.

.. code-block:: shell

  python -m lmp.script.train_model Elman-Net \
    --batch_size 32 \
    --beta1 0.9 \
    --beta2 0.999 \
    --ckpt_step 1000 \
    --d_emb 100 \
    --d_hid 100 \
    --dset_name wiki-text-2 \
    --eps 1e-8 \
    --exp_name my_model_exp \
    --init_lower -0.1 \
    --init_upper 0.1 \
    --label_smoothing 0.0 \
    --log_step 500 \
    --lr 1e-3 \
    --max_norm 10 \
    --max_seq_len 32 \
    --n_lyr 1 \
    --p_emb 0.5 \
    --p_hid 0.1 \
    --stride 32 \
    --tknzr_exp_name my_tknzr_exp \
    --total_step 100000 \
    --ver train \
    --warmup_step 5000 \
    --weight_decay 0.0

One can reduce overfitting with the following ways:

- Increase ``--batch_size``.
  This increase sample variance and make model hard to optimize.
- Increase ``--weight_decay``.
  This increase L2 penalty and make model output differences small when given large variance input.
- Reduce model parameters (In :py:class:`~lmp.model.ElmanNet` this means reducing ``--d_emb`` or ``--d_hid``).
  This make model capacity low and hard to memorize all samples.
  Thus model is forced to learn and utilize patterns found on different samples.
- Use dropout (In :py:class:`~lmp.model.ElmanNet` this means increasing ``--p_emb`` or ``--p_hid``).
  Dropout is a way to perform models ensembling without the cost of training multiple model instances.
- Use label smoothing so that model is not optimized to predict exactly ``0`` or ``1``.
  This can be done by setting ``--label_smoothing`` to positive values.
- Use any combinations of tricks above.

.. code-block:: shell

  python -m lmp.script.train_model Elman-Net \
    --batch_size 32 \
    --beta1 0.9 \
    --beta2 0.999 \
    --ckpt_step 1000 \
    --d_emb 50 \
    --d_hid 50 \
    --dset_name wiki-text-2 \
    --eps 1e-8 \
    --exp_name my_model_exp \
    --init_lower -0.1 \
    --init_upper 0.1 \
    --label_smoothing 0.0 \
    --log_step 500 \
    --lr 1e-3 \
    --max_norm 10 \
    --max_seq_len 32 \
    --n_lyr 1 \
    --p_emb 0.5 \
    --p_hid 0.5 \
    --stride 32 \
    --tknzr_exp_name my_tknzr_exp \
    --total_step 10000 \
    --ver train \
    --warmup_step 5000 \
    --weight_decay 1e-1

We use :py:class:`torch.optim.AdamW` to perform optimization.
Use ``--beta1``, ``--beta2``, ``--eps``, ``--lr`` and ``--weight_decay`` to adjust optimization hyperparameters.
We also use ``--max_norm`` to perform gradient clipping which avoids gradient explosion.

.. code-block:: shell

  python -m lmp.script.train_model Elman-Net \
    --batch_size 32 \
    --beta1 0.95 \
    --beta2 0.98 \
    --ckpt_step 1000 \
    --d_emb 100 \
    --d_hid 100 \
    --dset_name wiki-text-2 \
    --eps 1e-6 \
    --exp_name my_model_exp \
    --init_lower -0.1 \
    --init_upper 0.1 \
    --label_smoothing 0.0 \
    --log_step 500 \
    --lr 5e-4 \
    --max_norm 0.1 \
    --max_seq_len 32 \
    --n_lyr 1 \
    --p_emb 0.5 \
    --p_hid 0.1 \
    --stride 32 \
    --tknzr_exp_name my_tknzr_exp \
    --total_step 10000 \
    --ver train \
    --warmup_step 5000 \
    --weight_decay 0.0

You can use ``-h`` or ``--help`` options to get a list of available language models.

.. code-block:: shell

  python -m lmp.script.train_model -h

You can use ``-h`` or ``--help`` options on a specific language model to get a list of supported CLI arguments.

.. code-block:: shell

  python -m lmp.script.train_model Elman-Net -h
"""

import argparse
import copy
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

    group = model_subparser.add_argument_group('language model training hyperparameters')
    group.add_argument(
      '--batch_size',
      default=32,
      help='''
      Training mini-batch size.
      Default is ``32``.
      ''',
      type=int,
    )
    group.add_argument(
      '--beta1',
      default=0.9,
      help='''
      First beta coefficient of AdamW optimizer.
      Default is ``0.9``.
      ''',
      type=float,
    )
    group.add_argument(
      '--beta2',
      default=0.999,
      help='''
      Second beta coefficient of AdamW optimizer.
      Default is ``0.999``.
      ''',
      type=float,
    )
    group.add_argument(
      '--ckpt_step',
      default=1000,
      help='''
      Checkpoint save interval.
      Default is ``1000``.
      ''',
      type=int,
    )
    group.add_argument(
      '--dset_name',
      choices=lmp.dset.DSET_OPTS.keys(),
      default=lmp.dset.DemoDset.dset_name,
      help=f'''
      Name of the dataset which will be used to train language model.
      Default is ``{lmp.dset.DemoDset.dset_name}``.
      ''',
      type=str,
    )
    group.add_argument(
      '--eps',
      default=1e-8,
      help='''
      Denominator smooth term of AdamW optimizer.
      Default is ``1e-8``.
      ''',
      type=float,
    )
    group.add_argument(
      '--exp_name',
      default='my_model_exp',
      help='''
      Name of the language model training experiment.
      Default is ``my_model_exp``.
      ''',
      type=str,
    )
    group.add_argument(
      '--log_step',
      default=500,
      help='''
      Performance log interval.
      Default is ``500``.
      ''',
      type=int,
    )
    group.add_argument(
      '--lr',
      default=1e-3,
      help='''
      Learning rate.
      Default is ``1e-3``.
      ''',
      type=float,
    )
    group.add_argument(
      '--max_norm',
      default=10,
      help='''
      Gradient max-norm constraint.
      Default is ``10``.
      ''',
      type=float,
    )
    group.add_argument(
      '--max_seq_len',
      default=32,
      help='''
      Context window size.
      Default is ``32``.
      ''',
      type=int,
    )
    group.add_argument(
      '--seed',
      default=42,
      help='''
      Random seed.
      Default is ``42``.
      ''',
      type=int,
    )
    group.add_argument(
      '--stride',
      default=32,
      help='''
      Number of overlapping tokens between context windows.
      Default is ``32``.
      ''',
      type=int,
    )
    group.add_argument(
      '--tknzr_exp_name',
      default='my_tknzr_exp',
      help='''
      Name of the pre-trained tokenizer experiment.
      Default is ``my_tknzr_exp``.
      ''',
      type=str,
    )
    group.add_argument(
      '--total_step',
      default=10000,
      help='''
      Number of training steps.
      Default is ``10000``.
      ''',
      type=int,
    )
    group.add_argument(
      '--ver',
      default=None,
      help='''
      Version of the dataset.
      Default is ``None``.
      ''',
      type=str,
    )
    group.add_argument(
      '--warmup_step',
      default=5000,
      help='''
      Learning rate warm up steps.
      Default is ``5000``.
      ''',
      type=int,
    )
    group.add_argument(
      '--weight_decay',
      default=0.0,
      help='''
      Weight decay coefficient of AdamW optimizer.
      Default is ``0.0``.
      ''',
      type=float,
    )

    # Add model specific arguments.
    model_type.add_CLI_args(parser=model_subparser)

  args = parser.parse_args(argv)

  # `args.batch_size` validation.
  lmp.util.validate.raise_if_wrong_ordered(vals=[1, args.batch_size], val_names=['1', 'args.batch_size'])
  # `args.ckpt_step` validation.
  lmp.util.validate.raise_if_wrong_ordered(vals=[1, args.ckpt_step], val_names=['1', 'args.ckpt_step'])
  # `args.log_step` validation.
  lmp.util.validate.raise_if_wrong_ordered(vals=[1, args.log_step], val_names=['1', 'args.log_step'])
  # `args.max_norm` validation.
  lmp.util.validate.raise_if_wrong_ordered(vals=[0, args.max_norm], val_names=['0', 'args.max_norm'])

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

  # Convert dataset to language model training format.
  lm_format_dset = lmp.util.dset.LMFormatDset(
    dset=dset,
    max_seq_len=args.max_seq_len,
    stride=args.stride,
    tknzr=tknzr,
  )

  # Mini-batch random sampler.
  data_loader = torch.utils.data.DataLoader(
    batch_size=args.batch_size,
    dataset=lm_format_dset,
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
    weight_decay=args.weight_decay,
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
    for batch_cur_tkids, batch_next_tkids in data_loader:
      # Calculate next token prediction loss.
      loss, _ = model.cal_loss(
        batch_cur_tkids=batch_cur_tkids.to(device),
        batch_next_tkids=batch_next_tkids.to(device),
        batch_prev_states=None,
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

  # Save last checkpoint.
  lmp.util.model.save(ckpt=step, exp_name=args.exp_name, model=copy.deepcopy(model).to('cpu'))

  # Close tensorboard logger.
  writer.flush()
  writer.close()

  # Close CLI logger.
  cli_logger.close()


if __name__ == '__main__':
  main(argv=sys.argv[1:])
