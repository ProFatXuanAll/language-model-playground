"""Test parsing :py:class:`lmp.model.LSTM2000` arguments.

Test target:
- :py:meth:`lmp.model.LSTM2000.add_CLI_args`
- :py:meth:`lmp.script.ddp_train_model.parse_args`.
"""

import math

import lmp.script.ddp_train_model
import lmp.script.train_model
from lmp.dset import ALL_DSETS
from lmp.model import LSTM2000


def test_elman_net_parse_results(
  batch_size: int,
  beta1: float,
  beta2: float,
  ckpt_step: int,
  d_blk: int,
  d_emb: int,
  eps: float,
  exp_name: str,
  host_name: str,
  host_port: int,
  is_dset_in_memory: bool,
  log_step: int,
  lr: float,
  max_norm: float,
  max_seq_len: int,
  n_blk: int,
  n_epoch: int,
  n_worker: int,
  p_emb: float,
  p_hid: float,
  seed: int,
  tknzr_exp_name: str,
  warmup_step: int,
  wd: float,
  world_size: int,
) -> None:
  """Must correctly parse all arguments for :py:class:`lmp.model.LSTM2000`."""
  local_rank = 0
  rank = 0

  for dset_type in ALL_DSETS:
    for ver in dset_type.vers:
      argv = [
        LSTM2000.model_name,
        '--batch_size',
        str(batch_size),
        '--beta1',
        str(beta1),
        '--beta2',
        str(beta2),
        '--ckpt_step',
        str(ckpt_step),
        '--d_blk',
        str(d_blk),
        '--d_emb',
        str(d_emb),
        '--dset_name',
        dset_type.dset_name,
        '--eps',
        str(eps),
        '--exp_name',
        exp_name,
        '--log_step',
        str(log_step),
        '--lr',
        str(lr),
        '--max_norm',
        str(max_norm),
        '--max_seq_len',
        str(max_seq_len),
        '--n_blk',
        str(n_blk),
        '--n_epoch',
        str(n_epoch),
        '--n_worker',
        str(n_worker),
        '--p_emb',
        str(p_emb),
        '--p_hid',
        str(p_hid),
        '--seed',
        str(seed),
        '--tknzr_exp_name',
        str(tknzr_exp_name),
        '--ver',
        ver,
        '--warmup_step',
        str(warmup_step),
        '--wd',
        str(wd),
      ]

      if is_dset_in_memory:
        argv.append('--is_dset_in_memory')

      common_args = lmp.script.train_model.parse_args(argv=argv)

      argv.append('--host_name')
      argv.append(host_name)
      argv.append('--host_port')
      argv.append(str(host_port))
      argv.append('--local_rank')
      argv.append(local_rank)
      argv.append('--rank')
      argv.append(rank)
      argv.append('--world_size')
      argv.append(str(world_size))

      args = lmp.script.ddp_train_model.parse_args(argv=argv)

      for k in common_args.__dict__.keys():
        assert k in args.__dict__, \
          'Distributed parallel training script must support all arguments of original training script'

        if isinstance(common_args.__dict__[k], float):
          assert math.isclose(common_args.__dict__[k], args.__dict__[k]), 'Inconsistent arguments.'
        else:
          assert common_args.__dict__[k] == args.__dict__[k], 'Inconsistent arguments.'

      assert args.batch_size == batch_size
      assert math.isclose(args.beta1, beta1)
      assert math.isclose(args.beta2, beta2)
      assert args.ckpt_step == ckpt_step
      assert args.d_blk == d_blk
      assert args.d_emb == d_emb
      assert args.dset_name == dset_type.dset_name
      assert math.isclose(args.eps, eps)
      assert args.exp_name == exp_name
      assert args.host_name == host_name
      assert args.host_port == host_port
      assert args.is_dset_in_memory == is_dset_in_memory
      assert args.local_rank == local_rank
      assert args.log_step == log_step
      assert math.isclose(args.lr, lr)
      assert math.isclose(args.max_norm, max_norm)
      assert args.max_seq_len == max_seq_len
      assert args.model_name == LSTM2000.model_name
      assert args.n_blk == n_blk
      assert args.n_epoch == n_epoch
      assert args.n_worker == n_worker
      assert math.isclose(args.p_emb, p_emb)
      assert math.isclose(args.p_hid, p_hid)
      assert args.rank == rank
      assert args.seed == seed
      assert args.tknzr_exp_name == tknzr_exp_name
      assert args.ver == ver
      assert args.warmup_step == warmup_step
      assert math.isclose(args.wd, wd)
      assert args.world_size == world_size
