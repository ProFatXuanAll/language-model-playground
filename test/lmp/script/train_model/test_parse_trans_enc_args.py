"""Test parsing :py:class:`lmp.model.TransEnc` arguments.

Test target:
- :py:meth:`lmp.model.TransEnc.add_CLI_args`
- :py:meth:`lmp.script.train_model.parse_args`.
"""

import math

import lmp.script.train_model
from lmp.dset import ALL_DSETS, DemoDset
from lmp.model import TransEnc


def test_default_values() -> None:
  """Ensure default values consistency."""
  batch_size = 32
  beta1 = 0.9
  beta2 = 0.999
  ckpt_step = 1000
  d_ff = 1
  d_k = 1
  d_model = 1
  d_v = 1
  dset_name = DemoDset.dset_name
  eps = 1e-8
  exp_name = 'my_model_exp'
  init_lower = -0.1
  init_upper = 0.1
  label_smoothing = 0.0
  log_step = 500
  lr = 1e-3
  max_norm = 10
  max_seq_len = 32
  n_head = 1
  n_lyr = 1
  p = 0.0
  seed = 42
  stride = 32
  tknzr_exp_name = 'my_tknzr_exp'
  total_step = 10000
  ver = DemoDset.df_ver
  warmup_step = 5000
  weight_decay = 0.0

  argv = [TransEnc.model_name]
  args = lmp.script.train_model.parse_args(argv=argv)

  assert args.batch_size == batch_size
  assert math.isclose(args.beta1, beta1)
  assert math.isclose(args.beta2, beta2)
  assert args.ckpt_step == ckpt_step
  assert args.d_ff == d_ff
  assert args.d_k == d_k
  assert args.d_model == d_model
  assert args.d_v == d_v
  assert args.dset_name == dset_name
  assert math.isclose(args.eps, eps)
  assert args.exp_name == exp_name
  assert math.isclose(args.init_lower, init_lower)
  assert math.isclose(args.init_upper, init_upper)
  assert math.isclose(args.label_smoothing, label_smoothing)
  assert args.log_step == log_step
  assert math.isclose(args.lr, lr)
  assert math.isclose(args.max_norm, max_norm)
  assert args.max_seq_len == max_seq_len
  assert args.model_name == TransEnc.model_name
  assert args.n_head == n_head
  assert args.n_lyr == n_lyr
  assert math.isclose(args.p, p)
  assert args.seed == seed
  assert args.stride == stride
  assert args.tknzr_exp_name == tknzr_exp_name
  assert args.total_step == total_step
  assert args.ver == ver
  assert args.warmup_step == warmup_step
  assert math.isclose(args.weight_decay, weight_decay)


def test_trans_enc_parse_results(
  batch_size: int,
  beta1: float,
  beta2: float,
  ckpt_step: int,
  d_ff: int,
  d_k: int,
  d_model: int,
  d_v: int,
  eps: float,
  exp_name: str,
  init_lower: float,
  init_upper: float,
  label_smoothing: float,
  log_step: int,
  lr: float,
  max_norm: float,
  max_seq_len: int,
  n_head: int,
  n_lyr: int,
  p_hid: float,
  seed: int,
  stride: int,
  tknzr_exp_name: str,
  total_step: int,
  warmup_step: int,
  weight_decay: float,
) -> None:
  """Must correctly parse all arguments for :py:class:`lmp.model.TransEnc`."""
  for dset_type in ALL_DSETS:
    for ver in dset_type.vers:
      argv = [
        TransEnc.model_name,
        '--batch_size',
        str(batch_size),
        '--beta1',
        str(beta1),
        '--beta2',
        str(beta2),
        '--ckpt_step',
        str(ckpt_step),
        '--d_ff',
        str(d_ff),
        '--d_k',
        str(d_k),
        '--d_model',
        str(d_model),
        '--d_v',
        str(d_v),
        '--dset_name',
        dset_type.dset_name,
        '--eps',
        str(eps),
        '--exp_name',
        exp_name,
        '--init_lower',
        str(init_lower),
        '--init_upper',
        str(init_upper),
        '--label_smoothing',
        str(label_smoothing),
        '--log_step',
        str(log_step),
        '--lr',
        str(lr),
        '--max_norm',
        str(max_norm),
        '--max_seq_len',
        str(max_seq_len),
        '--n_head',
        str(n_head),
        '--n_lyr',
        str(n_lyr),
        '--p',
        str(p_hid),
        '--seed',
        str(seed),
        '--stride',
        str(stride),
        '--tknzr_exp_name',
        str(tknzr_exp_name),
        '--total_step',
        str(total_step),
        '--ver',
        ver,
        '--warmup_step',
        str(warmup_step),
        '--weight_decay',
        str(weight_decay),
      ]

      args = lmp.script.train_model.parse_args(argv=argv)

      assert args.batch_size == batch_size
      assert math.isclose(args.beta1, beta1)
      assert math.isclose(args.beta2, beta2)
      assert args.ckpt_step == ckpt_step
      assert args.d_ff == d_ff
      assert args.d_k == d_k
      assert args.d_model == d_model
      assert args.d_v == d_v
      assert args.dset_name == dset_type.dset_name
      assert math.isclose(args.eps, eps)
      assert args.exp_name == exp_name
      assert math.isclose(args.init_lower, init_lower)
      assert math.isclose(args.init_upper, init_upper)
      assert math.isclose(args.label_smoothing, label_smoothing)
      assert args.log_step == log_step
      assert math.isclose(args.lr, lr)
      assert math.isclose(args.max_norm, max_norm)
      assert args.max_seq_len == max_seq_len
      assert args.model_name == TransEnc.model_name
      assert args.n_head == n_head
      assert args.n_lyr == n_lyr
      assert math.isclose(args.p, p_hid)
      assert args.seed == seed
      assert args.stride == stride
      assert args.tknzr_exp_name == tknzr_exp_name
      assert args.total_step == total_step
      assert args.ver == ver
      assert args.warmup_step == warmup_step
      assert math.isclose(args.weight_decay, weight_decay)
