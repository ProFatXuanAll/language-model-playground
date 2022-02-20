r"""Use this script to distributedly train language model on particular dataset.

This script is the distributed version of :doc:`lmp.script.train_model </script/train_model>`.  CLI arguments are the
same as :doc:`lmp.script.train_model </script/train_model>` except for the addtional distributed training arguments.

See Also
--------
:doc:`lmp.script.train_model </script/train_model>`
  Language model training script.

Examples
--------
The following example script use two process on localhost ``127.0.0.1`` to train Elman Net model
:py:class:`lmp.model.ElmanNet` on Wiki-Text-2 dataset :py:class:`lmp.dset.WikiText2Dset` with ``train`` version.

.. code-block:: shell

   # process 0 on localhost
   python -m lmp.script.d_train_model Elman-Net \
     --batch_size 32 \
     --beta1 0.9 \
     --beta2 0.99 \
     --ckpt_step 1000 \
     --d_emb 100 \
     --d_hid 100 \
     --dset_name wiki-text-2 \
     --eps 1e-8 \
     --exp_name my_model_exp \
     --host_name 127.0.0.1 \
     --host_port 30678 \
     --local_rank 0 \
     --log_step 200 \
     --lr 1e-4 \
     --max_norm 1 \
     --max_seq_len 128 \
     --n_epoch 10 \
     --p_emb 0.5 \
     --p_hid 0.1 \
     --rank 0 \
     --tknzr_exp_name my_tknzr_exp \
     --ver train \
     --warmup_step 10000 \
     --wd 1e-2 \
     --world_size 2

   # process 1 on localhost
   python -m lmp.script.d_train_model Elman-Net \
     --batch_size 32 \
     --beta1 0.9 \
     --beta2 0.99 \
     --ckpt_step 1000 \
     --d_emb 100 \
     --d_hid 100 \
     --dset_name wiki-text-2 \
     --eps 1e-8 \
     --exp_name my_model_exp \
     --host_name 127.0.0.1 \
     --host_port 30678 \
     --local_rank 1 \
     --log_step 200 \
     --lr 1e-4 \
     --max_norm 1 \
     --max_seq_len 128 \
     --n_epoch 10 \
     --p_emb 0.5 \
     --p_hid 0.1 \
     --rank 1 \
     --tknzr_exp_name my_tknzr_exp \
     --ver train \
     --warmup_step 10000 \
     --wd 1e-2 \
     --world_size 2

You can use ``-h`` or ``--help`` options to get a list of available language models.

.. code-block:: shell

   python -m lmp.script.d_train_model -h

You can use ``-h`` or ``--help`` options on a specific language model to get a list of supported CLI arguments.

.. code-block:: shell

   python -m lmp.script.d_train_model Elman-Net -h
"""

import argparse
import gc
import sys
from datetime import timedelta
from typing import Final, List

import torch
import torch.distributed as dist
import torch.nn.parallel
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

HOST_RANK: Final[int] = 0


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

    group = model_subparser.add_argument_group('distributed training arguments')
    group.add_argument(
      '--host_name',
      help='Host name of distributed training main process.',
      required=True,
      type=str,
    )
    group.add_argument(
      '--host_port',
      help='Listening port of distributed training main process.',
      required=True,
      type=int,
    )
    group.add_argument(
      '--local_rank',
      help='CUDA device to be used by the current process.',
      required=True,
      type=int,
    )
    group.add_argument(
      '--rank',
      help='Rank of the current process.',
      required=True,
      type=int,
    )
    group.add_argument(
      '--world_size',
      help='Number of process to perform distributed training.',
      required=True,
      type=int,
    )

    # Training required arguments.
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
      '--n_epoch',
      help='Number of training epochs.',
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
      help='Random seed.',
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

  # Save training configuration.
  lmp.util.cfg.save(args=args, exp_name=args.exp_name)

  # We use TCP to perform RPC.  Timeout is set to 5 minutes.
  store = dist.TCPStore(
    is_master=args.rank == HOST_RANK,
    host_name=args.host_name,
    port=args.host_port,
    timeout=timedelta(minutes=5),
    world_size=args.world_size,
  )

  # Use NCCL backend to perform CUDA collectives.
  dist.init_process_group(
    backend=dist.Backend.NCCL,
    store=store,
    rank=args.rank,
    timeout=timedelta(minutes=5),
    world_size=args.world_size,
  )

  # Sync arguments.
  dist_args_k = ['host_name', 'host_port', 'local_rank', 'rank', 'world_size']
  for k in args.__dict__.keys():
    if k in dist_args_k:
      continue

    # Host broadcast arguments.
    if args.rank == HOST_RANK:
      store.set(k, str(args.__dict__[k]))
    # Non-host receive host arguments.
    else:
      v = store.get(k)
      if isinstance(args.__dict__[k], str):
        args.__dict__[k] = v.decode('utf-8')
      else:
        args.__dict__[k] = type(args.__dict__[k])(v)

  train(args=args)


def train(args: argparse.Namespace) -> None:
  """Train language model."""
  # Set random seed for reproducibility.  Note that each process use different seed to get different slice of batch.
  lmp.util.rand.set_seed(seed=args.seed + args.rank)

  # Get dataset instance with specified version.
  dset = lmp.util.dset.load(**args.__dict__)

  # Mini-batch sampler.  Each process will get batches exclusive to itself.
  dist_sampler = torch.utils.data.distributed.DistributedSampler(
    num_replicas=args.world_size,
    rank=args.rank,
    dataset=dset,
    shuffle=True,
  )

  # Create dataloader with distributed sampler.
  data_loader = torch.utils.data.DataLoader(
    dataset=dset,
    batch_size=args.batch_size // args.world_size,
    sampler=dist_sampler,
  )

  # Load pre-trained tokenizer.
  tknzr = lmp.util.tknzr.load(exp_name=args.tknzr_exp_name)

  # Get model running device.
  device = torch.device('cpu')
  if torch.cuda.is_available():
    device = torch.device(f'cuda:{args.local_rank}')

  # Get new model instance and move model to running device.
  model = lmp.util.model.create(tknzr=tknzr, **args.__dict__)
  model = model.train()
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
    total_step=args.n_epoch * len(data_loader),
    warmup_step=args.warmup_step,
  )

  # Create DPP model.
  model = torch.nn.parallel.DistributedDataParallel(model)

  # Get tensorboard logger instance.  Only main process need to log performance.
  if args.rank == HOST_RANK:
    writer = lmp.util.log.get_tb_logger(exp_name=args.exp_name)
  else:
    writer = None

  # Log performance target.
  pre_avg_loss = 0.0
  avg_loss = 0.0

  # Global optimization step.
  step = 0
  for epoch in range(args.n_epoch):
    # Update random sample order.
    dist_sampler.set_epoch(epoch)

    # Processes can have unevenly distributed number of batch.  Thus one must use join to avoid dead lock.
    with model.join():
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

        # Update learning rate.
        schdl.step()

        # Clean up gradient.
        optim.zero_grad()

        # Increment global step.
        step += 1

        # Save checkpoint for each `ckpt_step` step.  Only main process need to save checkpoint.
        if args.rank == HOST_RANK and step % args.ckpt_step == 0:
          lmp.util.model.save(ckpt=step, exp_name=args.exp_name, model=model)

        # Log performance for each `log_step` step.
        if step % args.log_step == 0:
          avg_loss = avg_loss / args.log_step

          # Log on CLI.
          tqdm_data_loader.set_description(f'epoch: {epoch}, loss: {avg_loss:.6f}')

          # Log on tensorboard.  Only main process need to log performance.
          if args.rank == HOST_RANK:
            writer.add_scalar(f'train-loss/{args.dset_name}/{args.ver}', avg_loss, step)
            writer.add_scalar('lr', schdl.get_last_lr()[0], step)

          # Refresh log performance.
          pre_avg_loss = avg_loss
          avg_loss = 0.0

  # Save last checkpoint.  Only main process need to save checkpoint.
  if args.rank == HOST_RANK:
    lmp.util.model.save(ckpt=step, exp_name=args.exp_name, model=model)

    # Close tensorboard logger.
    writer.close()

  # Free memory.  This is only need for unit test.
  del args
  del avg_loss
  del data_loader
  del device
  del dset
  del tknzr
  del model
  del optim
  del pre_avg_loss
  del schdl
  del step
  del tqdm_data_loader
  del writer
  torch.cuda.empty_cache()
  gc.collect()


if __name__ == '__main__':
  main(argv=sys.argv[1:])
