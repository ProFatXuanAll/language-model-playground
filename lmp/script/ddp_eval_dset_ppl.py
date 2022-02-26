r"""Distributedly evaluate language model checkpoints on multiple processes / nodes by data parallism.

This script is distributed data parallel version of :doc:`lmp.script.eval_dset_ppl </script/eval_dset_ppl>`.  Other
than distributed evaluation setup CLI arguments, the rest arguments are the same as
:doc:`lmp.script.eval_dset_ppl </script/eval_dset_ppl>`.

Note
----
To use this script across multiple nodes, each node must have model configuration and all model checkpoints.

See Also
--------
:doc:`lmp.script.eval_dset_ppl </script/eval_dset_ppl>`
  Language model evaluation script.

Examples
--------
The following example script use two process on localhost ``127.0.0.1`` to evaluation experiment ``my_model_exp`` on
Wiki-Text-2 dataset :py:class:`lmp.dset.WikiText2Dset` with ``valid`` version.

.. code-block:: shell

   # process 0 on localhost
   python -m lmp.script.ddp_eval_dset_ppl wiki-text-2 \
     --batch_size 64 \
     --first_ckpt 0 \
     --host_name 127.0.0.1 \
     --host_port 30678 \
     --last_ckpt -1 \
     --local_rank 0 \
     --exp_name my_model_exp \
     --rank 0 \
     --ver valid \
     --world_size 2

   # process 1 on localhost
   python -m lmp.script.ddp_eval_dset_ppl wiki-text-2 \
     --batch_size 64 \
     --first_ckpt 0 \
     --host_name 127.0.0.1 \
     --host_port 30678 \
     --last_ckpt -1 \
     --local_rank 1 \
     --exp_name my_model_exp \
     --rank 1 \
     --ver valid \
     --world_size 2

You can use ``-h`` or ``--help`` options to get a list of supported CLI arguments.

.. code-block:: shell

   python -m lmp.script.ddp_eval_dset_ppl -h
"""

import argparse
import gc
import os
import sys
from datetime import timedelta
from typing import Final, List

import torch
import torch.distributed as dist
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

    group = dset_subparser.add_argument_group('distributed evaluation arguments')
    group.add_argument(
      '--host_name',
      help='Host name of distributed evaluation main process.',
      required=True,
      type=str,
    )
    group.add_argument(
      '--host_port',
      help='Listening port of distributed evaluation main process.',
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

    # Evaluation required arguments.
    group = dset_subparser.add_argument_group('language model evaluation arguments')
    group.add_argument(
      '--batch_size',
      help='Evaluation mini-batch size.',
      required=True,
      type=int,
    )
    group.add_argument(
      '--exp_name',
      help='Pre-trained language model experiment name.',
      required=True,
      type=str,
    )
    group.add_argument(
      '--first_ckpt',
      help='The first checkpoint of pre-trained language model to be evaluated.',
      required=True,
      type=int,
    )

    # Optional arguments.
    group.add_argument(
      '--is_dset_in_memory',
      action='store_true',
      help='If set to true, then the whole dataset will be loaded in memory.  This will speed up text preprocessing.  '
      'Default is ``False``.'
    )
    group.add_argument(
      '--last_ckpt',
      default=-1,
      help='The last checkpoint of pre-trained language model to be evaluated.  Default is ``-1``.',
      type=int,
    )
    group.add_argument(
      '--n_worker',
      default=0,
      help='Number of workers (processes) to use to preprocess text.  We recommand to set to ``0`` when your '
      'mini-batch size is less than ``256``, set to ``4`` otherwise.  Default is ``0``.',
      type=int,
    )
    group.add_argument(
      '--seed',
      default=42,
      help='Random seed.  Default is ``42``',
      type=int,
    )
    group.add_argument(
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

  # `args.batch_size` validation.
  lmp.util.validate.raise_if_wrong_ordered(vals=[1, args.batch_size], val_names=['1', 'args.batch_size'])
  # `args.first_ckpt` validation.
  lmp.util.validate.raise_if_wrong_ordered(vals=[-1, args.first_ckpt], val_names=['-1', 'args.first_ckpt'])
  # `args.last_ckpt` validation.
  lmp.util.validate.raise_if_wrong_ordered(vals=[-1, args.last_ckpt], val_names=['-1', 'args.last_ckpt'])
  # `args.n_worker` validation.
  lmp.util.validate.raise_if_wrong_ordered(
    vals=[0, args.n_worker, len(os.sched_getaffinity(0))],
    val_names=['0', 'args.n_worker', 'number of available CPUs'],
  )
  lmp.util.validate.raise_if_wrong_ordered(
    vals=[args.n_worker, args.batch_size],
    val_names=['args.n_worker', 'args.batch_size'],
  )

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

  # Set random seed for reproducibility.  Note that each process use different seed to get different slice of batch.
  lmp.util.rand.set_seed(seed=args.seed + args.rank)

  # Get model running device.
  device = torch.device('cpu')
  if torch.cuda.is_available():
    device = torch.device(f'cuda:{args.local_rank}')

  # Load pre-trained model configuration.
  model_cfg = lmp.util.cfg.load(exp_name=args.exp_name)

  # Load pre-trained tokenizer instance.
  tknzr = lmp.util.tknzr.load(exp_name=model_cfg.tknzr_exp_name)

  # Get dataset instance and convert samples to tensor.
  if args.is_dset_in_memory:
    dset: torch.utils.data.Dataset = lmp.util.dset.FastTensorDset(
      dset=lmp.util.dset.load(**args.__dict__),
      max_seq_len=model_cfg.max_seq_len,
      tknzr=tknzr,
    )
  else:
    dset = lmp.util.dset.SlowTensorDset(
      dset=lmp.util.dset.load(**args.__dict__),
      max_seq_len=model_cfg.max_seq_len,
      tknzr=tknzr,
    )

  dset_size = len(dset)

  # Mini-batch sampler.  Each process will get batches exclusive to itself.
  dist_sampler = torch.utils.data.distributed.DistributedSampler(
    num_replicas=args.world_size,
    rank=args.rank,
    dataset=dset,
    shuffle=False,
  )

  # Mini-batch distributed random sampler.  Only when `args.n_worker > 0` we set `persisten_worker = True`.  We set
  # `pin_memory = True` to speed up process (which only speed up a few seconds).
  data_loader = torch.utils.data.DataLoader(
    batch_size=args.batch_size // args.world_size,
    dataset=dset,
    num_workers=args.n_worker,
    persistent_workers=bool(args.n_worker != 0),
    pin_memory=True,
    sampler=dist_sampler,
  )

  # Get tensorboard logger instance.  Only main process need to log performance.
  if args.rank == HOST_RANK:
    writer = lmp.util.log.get_tb_logger(exp_name=args.exp_name)
  else:
    writer = None

  # Evaluate checkpoints within ranges.
  for ckpt in lmp.util.model.list_ckpts(exp_name=args.exp_name, first_ckpt=args.first_ckpt, last_ckpt=args.last_ckpt):
    # Load pre-trained model instance.
    model = lmp.util.model.load(ckpt=ckpt, exp_name=args.exp_name)

    # Set model to evaluation model.  This turn off dropout layers in model.
    model = model.eval()

    # Move model to running device.
    model = model.to(device)

    # Create DDP model.
    dpp_model = torch.nn.parallel.DistributedDataParallel(model)

    # Processes can have unevenly distributed number of batch.  Thus one must use `ddp_model.join()` to avoid dead lock.
    with dpp_model.join():
      # Record average perplexity.
      avg_ppl = 0.0
      for batch_tkids in tqdm(data_loader):
        # Encode text into token ids.  We convert token ids into tensor and move to the same running device as model.
        batch_tkids = batch_tkids.to(device)

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
        batch_ppl = lmp.util.metric.ppl(batch_tkids=batch_next_tkids, batch_tkids_pd=torch.stack(batch_tkids_pd, dim=1))

        # Sum `batch_ppl` from each process.
        dist.all_reduce(batch_ppl, op=dist.ReduceOp.SUM)

        # Accumulate average perplexity.
        avg_ppl += (batch_ppl / dset_size).sum().item()

    # Log average perplexity on dataset to CLI and tensorboard.  Only main process need to log performance.
    if args.rank == HOST_RANK:
      writer.add_scalar(f'ppl/{args.dset_name}/{args.ver}', avg_ppl, ckpt)
    print(f'checkpoint: {ckpt}, avg ppl: {avg_ppl}')

  # Free memory.  This is only need for unit test.
  del args
  del avg_ppl
  del batch_cur_tkids
  del batch_next_tkids
  del batch_next_tkids_pd
  del batch_ppl
  del batch_prev_states
  del batch_tkids
  del batch_tkids_pd
  del ckpt
  del data_loader
  del device
  del dset
  del dset_size
  del model
  del model_cfg
  del tknzr
  del writer
  torch.cuda.empty_cache()
  gc.collect()


if __name__ == '__main__':
  main(argv=sys.argv[1:])
