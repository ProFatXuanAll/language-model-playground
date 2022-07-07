"""Test parsing :py:class:`lmp.model.LSTM2002` arguments.

Test target:
- :py:meth:`lmp.model.LSTM2002.add_CLI_args`
- :py:meth:`lmp.script.train_model.parse_args`.
"""

import math

import lmp.script.train_model
from lmp.dset import ALL_DSETS
from lmp.model import LSTM2002


def test_lstm_2002_parse_results(
  batch_size: int,
  beta1: float,
  beta2: float,
  ckpt_step: int,
  ctx_win: int,
  d_blk: int,
  d_emb: int,
  eps: float,
  exp_name: str,
  log_step: int,
  lr: float,
  max_norm: float,
  max_seq_len: int,
  n_blk: int,
  n_epoch: int,
  p_emb: float,
  p_hid: float,
  seed: int,
  tknzr_exp_name: str,
  warmup_step: int,
  wd: float,
) -> None:
  """Must correctly parse all arguments for :py:class:`lmp.model.LSTM2002`."""
  for dset_type in ALL_DSETS:
    for ver in dset_type.vers:
      argv = [
        LSTM2002.model_name,
        '--batch_size',
        str(batch_size),
        '--beta1',
        str(beta1),
        '--beta2',
        str(beta2),
        '--ckpt_step',
        str(ckpt_step),
        '--ctx_win',
        str(ctx_win),
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

      args = lmp.script.train_model.parse_args(argv=argv)

      assert args.batch_size == batch_size
      assert math.isclose(args.beta1, beta1)
      assert math.isclose(args.beta2, beta2)
      assert args.ckpt_step == ckpt_step
      assert args.ctx_win == ctx_win
      assert args.d_blk == d_blk
      assert args.d_emb == d_emb
      assert args.dset_name == dset_type.dset_name
      assert math.isclose(args.eps, eps)
      assert args.exp_name == exp_name
      assert args.log_step == log_step
      assert math.isclose(args.lr, lr)
      assert math.isclose(args.max_norm, max_norm)
      assert args.max_seq_len == max_seq_len
      assert args.model_name == LSTM2002.model_name
      assert args.n_blk == n_blk
      assert args.n_epoch == n_epoch
      assert math.isclose(args.p_emb, p_emb)
      assert math.isclose(args.p_hid, p_hid)
      assert args.seed == seed
      assert args.tknzr_exp_name == tknzr_exp_name
      assert args.ver == ver
      assert args.warmup_step == warmup_step
      assert math.isclose(args.wd, wd)
