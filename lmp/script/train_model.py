r"""Train language model.

Use this script to train language model on particular dataset.  This script is usually run after training tokenizer.
Training performance will be shown on both CLI and tensorboard.  Use ``pipenv run tensorboard`` to launch tensorboard
and open browser with URL http://localhost:6006/ to see training performance.

See Also
--------
lmp.model
  All available language models.
lmp.script.train_tknzr
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
     --d_emb 100 \
     --dset_name wiki-text-2 \
     --eps 1e-8 \
     --exp_name my_model_exp \
     --log_step 200 \
     --lr 1e-4 \
     --max_norm 1 \
     --max_seq_len 128 \
     --n_epoch 10 \
     --tknzr_exp_name my_tknzr_exp \
     --ver train \
     --wd 1e-2

The training result will be save at path ``root/exp/my_model_exp`` and can be reused by other scripts.  Here ``root``
refers to :py:attr:`lmp.util.path.PROJECT_ROOT`.  We only save checkpoints for each ``--ckpt_step`` step and log
performance for each ``--log_step`` step.

One can increase ``--n_epoch`` to train more epochs.  Be careful model might overfit on datasets if model were trained
with too many epochs.

.. code-block:: shell

   python -m lmp.script.train_model Elman-Net \
     --batch_size 32 \
     --beta1 0.9 \
     --beta2 0.99 \
     --ckpt_step 1000 \
     --d_emb 100 \
     --dset_name wiki-text-2 \
     --eps 1e-8 \
     --exp_name my_model_exp \
     --log_step 200 \
     --lr 1e-4 \
     --max_norm 1 \
     --max_seq_len 128 \
     --n_epoch 20 \
     --tknzr_exp_name my_tknzr_exp \
     --ver train \
     --wd 1e-2

One can reduce overfitting with the following ways:

- Increase ``--batch_size``.  This increase sample variance and make model hard to optimize.
- Increase ``--wd``.  This increase L2 penalty and make model output differences small when given large variance input.
- Reduce model parameters (In :py:class:`lmp.model.ElmanNet` this means reducing ``--d_emb``).  This make model
  capacity low and hard to memorize all samples.  Thus model is forced to learn and utilize patterns found on different
  samples.
- Use dropout.  Dropout is a way to perform models ensembling without the cost of training multiple model instances.
- Use any combinations of tricks above.

.. code-block:: shell

   python -m lmp.script.train_model Elman-Net \
     --batch_size 32 \
     --beta1 0.9 \
     --beta2 0.99 \
     --ckpt_step 1000 \
     --d_emb 50 \
     --dset_name wiki-text-2 \
     --eps 1e-8 \
     --exp_name my_model_exp \
     --log_step 200 \
     --lr 1e-4 \
     --max_norm 1 \
     --max_seq_len 128 \
     --n_epoch 10 \
     --tknzr_exp_name my_tknzr_exp \
     --ver train \
     --wd 1e-1

We use :py:class:`torch.optim.AdamW` to perform optimization.  Use ``--beta1``, ``--beta2``, ``--eps``, ``--lr`` and
``--wd`` to adjust optimization hyperparameters.  We also use ``--max_norm`` to perform gradient clipping which avoid
gradient explosion.

.. code-block:: shell

   python -m lmp.script.train_model Elman-Net \
     --batch_size 32 \
     --beta1 0.95 \
     --beta2 0.98 \
     --ckpt_step 1000 \
     --dset_name wiki-text-2 \
     --eps 1e-6 \
     --exp_name my_model_exp \
     --log_step 200 \
     --lr 5e-4 \
     --max_norm 0.1 \
     --max_seq_len 128 \
     --n_epoch 10 \
     --tknzr_exp_name my_tknzr_exp \
     --ver train \
     --d_emb 100 \
     --wd 1e-2

You can use ``-h`` or ``--help`` options to get a list of available language models.

.. code-block:: shell

   python -m lmp.script.train_model -h

You can use ``-h`` or ``--help`` options on a specific language model to get a list of supported CLI arguments.

.. code-block:: shell

   python -m lmp.script.train_model Elman-Net -h
"""

import argparse
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
  parser = argparse.ArgumentParser('python -m lmp.script.train_model', description='Train language model.')

  # Use model name to create subparser for all language models.
  subparsers = parser.add_subparsers(dest='model_name', required=True)
  for model_name, model_type in lmp.model.MODEL_OPTS.items():
    model_subparser = subparsers.add_parser(model_name, description=f'Training {model_type.__name__} language model.')

    # Add model specific arguments.
    model_type.train_parser(parser=model_subparser)

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

  # Save training configuration.
  lmp.util.cfg.save(args=args, exp_name=args.exp_name)

  # Set random seed for reproducibility.
  lmp.util.rand.set_seed(seed=args.seed)

  # Get dataset instance with specified version.
  dset = lmp.util.dset.load(**args.__dict__)

  # Mini-batch random sampler.
  data_loader = torch.utils.data.DataLoader(batch_size=args.batch_size, dataset=dset, shuffle=True)

  # Load pre-trained tokenizer.
  tknzr_cfg = lmp.util.cfg.load(exp_name=args.tknzr_exp_name)
  tknzr = lmp.util.tknzr.load(exp_name=args.tknzr_exp_name, tknzr_name=tknzr_cfg.tknzr_name)

  # Get model running device.
  device = torch.device('cpu')
  if torch.cuda.is_available():
    device = torch.device('cuda')

  # Get new model instance and move model to running device.
  model = lmp.util.model.create(tknzr=tknzr, **args.__dict__)
  model = model.train()
  model = model.to(device)

  # Remove weight decay on bias and layer-norm.  This must be done only after moving model to running device.
  no_decay = ['bias', 'LayerNorm.weight']
  optim_group_params = [
    {
      'params': [param for name, param in model.named_parameters() if not any(nd in name for nd in no_decay)],
      'weight_decay': args.wd,
    },
    {
      'params': [param for name, param in model.named_parameters() if any(nd in name for nd in no_decay)],
      'weight_decay': 0.0,
    },
  ]

  # Get new optimizer instance.  We always use AdamW as our model optimizer.
  optim = torch.optim.AdamW(optim_group_params, betas=(args.beta1, args.beta2), eps=args.eps, lr=args.lr)

  # Get tensorboard logger instance.
  writer = lmp.util.log.get_tb_logger(exp_name=args.exp_name)

  # Log performance target.
  pre_avg_loss = 0.0
  avg_loss = 0.0

  # Global optimization step.
  step = 0
  for epoch in range(args.n_epoch):
    tqdm_data_loader = tqdm(data_loader, desc=f'epoch: {epoch}, loss: {pre_avg_loss:.6f}')
    for batch_txt in tqdm_data_loader:
      # Encode batch text into batch token ids.  We convert batch token ids into tensor and move to tensor to the same
      # running device as model.  Since CUDA only support integer with Long type, we use `torch.LongTensor` instead of
      # `torch.IntTensor`.
      batch_tkids = torch.LongTensor(tknzr.batch_enc(batch_txt=batch_txt, max_seq_len=args.max_seq_len)).to(device)

      # Format batch token ids to satisfy language model training format.
      batch_cur_tkids = batch_tkids[..., :-1]
      batch_next_tkids = batch_tkids[..., 1:]

      # Calculate loss using loss function.
      loss = model(batch_cur_tkids=batch_cur_tkids, batch_next_tkids=batch_next_tkids)

      # Accumulate average loss for logging.  Use `.item()` to avoid construct tensor graph.
      avg_loss += loss.item()

      # Perform backward pass / back propagation.
      loss.backward()

      # Perform gradient clipping to avoid gradient explosion.
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)

      # Gradient descent.
      optim.step()

      # Clean up gradient.
      optim.zero_grad()

      # Increment global step.
      step += 1

      # Save checkpoint for each `ckpt_step` step.
      if step % args.ckpt_step == 0:
        lmp.util.model.save(ckpt=step, exp_name=args.exp_name, model=model)

      # Log performance for each `log_step` step.
      if step % args.log_step == 0:
        avg_loss = avg_loss / args.log_step

        # Log on CLI.
        tqdm_data_loader.set_description(f'epoch: {epoch}, loss: {avg_loss:.6f}')

        # Log on tensorboard
        writer.add_scalar(f'train-loss/{args.dset_name}/{args.ver}', avg_loss, step)

        # Refresh log performance.
        pre_avg_loss = avg_loss
        avg_loss = 0.0

  # Save last checkpoint.
  lmp.util.model.save(ckpt=step, exp_name=args.exp_name, model=model)

  # Close tensorboard logger.
  writer.close()


if __name__ == '__main__':
  main(argv=sys.argv[1:])
