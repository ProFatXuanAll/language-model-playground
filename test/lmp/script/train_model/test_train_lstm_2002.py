"""Test training :py:class:`lmp.model.LSTM2002`.

Test target:
- :py:meth:`lmp.model.LSTM2002.forward`
- :py:meth:`lmp.script.train_model.main`.
"""

import math
import os
import re

import torch

import lmp.script.train_model
import lmp.util.cfg
import lmp.util.model
import lmp.vars
from lmp.dset import DemoDset
from lmp.model import LSTM2002


def test_train_lstm_2002_on_demo(
  batch_size: int,
  beta1: float,
  beta2: float,
  capsys,
  cfg_file_path: str,
  ckpt_dir_path: str,
  ckpt_step: int,
  d_blk: int,
  d_emb: int,
  eps: float,
  exp_name: str,
  init_fb: float,
  init_ib: float,
  init_lower: float,
  init_ob: float,
  init_upper: float,
  label_smoothing: float,
  log_step: int,
  lr: float,
  max_norm: float,
  max_seq_len: int,
  n_blk: int,
  n_lyr: int,
  p_emb: float,
  p_hid: float,
  seed: int,
  stride: int,
  tknzr_exp_name: str,
  total_step: int,
  train_log_dir_path: str,
  warmup_step: int,
  weight_decay: float,
) -> None:
  """Successfully train model :py:class:`lmp.model.LSTM2002` on :py:class:`lmp.dset.DemoDset` dataset."""
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
    '--d_blk',
    str(d_blk),
    '--d_emb',
    str(d_emb),
    '--dset_name',
    DemoDset.dset_name,
    '--eps',
    str(eps),
    '--exp_name',
    exp_name,
    '--init_fb',
    str(init_fb),
    '--init_ib',
    str(init_ib),
    '--init_lower',
    str(init_lower),
    '--init_ob',
    str(init_ob),
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
    '--n_blk',
    str(n_blk),
    '--n_lyr',
    str(n_lyr),
    '--p_emb',
    str(p_emb),
    '--p_hid',
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
    'test',  # Make training faster.
    '--warmup_step',
    str(warmup_step),
    '--weight_decay',
    str(weight_decay),
  ]

  lmp.script.train_model.main(argv=argv)

  # Must save training configuration.
  assert os.path.exists(cfg_file_path)
  # Must save model checkpoints.
  assert os.path.exists(ckpt_dir_path)
  # Must have at least one checkpoints.
  assert os.path.exists(os.path.join(ckpt_dir_path, f'model-{ckpt_step}.pt'))
  # Must log model performance.
  assert os.path.exists(train_log_dir_path)

  cfg = lmp.util.cfg.load(exp_name=exp_name)
  assert cfg.batch_size == batch_size
  assert math.isclose(cfg.beta1, beta1)
  assert math.isclose(cfg.beta2, beta2)
  assert cfg.ckpt_step == ckpt_step
  assert cfg.d_blk == d_blk
  assert cfg.d_emb == d_emb
  assert cfg.dset_name == DemoDset.dset_name
  assert math.isclose(cfg.eps, eps)
  assert cfg.exp_name == exp_name
  assert math.isclose(cfg.init_fb, init_fb)
  assert math.isclose(cfg.init_ib, init_ib)
  assert math.isclose(cfg.init_lower, init_lower)
  assert math.isclose(cfg.init_ob, init_ob)
  assert math.isclose(cfg.init_upper, init_upper)
  assert math.isclose(cfg.label_smoothing, label_smoothing)
  assert cfg.log_step == log_step
  assert math.isclose(cfg.lr, lr)
  assert math.isclose(cfg.max_norm, max_norm)
  assert cfg.max_seq_len == max_seq_len
  assert cfg.model_name == LSTM2002.model_name
  assert cfg.n_blk == n_blk
  assert cfg.n_lyr == n_lyr
  assert math.isclose(cfg.p_emb, p_emb)
  assert math.isclose(cfg.p_hid, p_hid)
  assert cfg.seed == seed
  assert cfg.stride == stride
  assert cfg.tknzr_exp_name == tknzr_exp_name
  assert cfg.total_step == total_step
  assert cfg.ver == 'test'
  assert cfg.warmup_step == warmup_step
  assert math.isclose(cfg.weight_decay, weight_decay)

  model = lmp.util.model.load(ckpt=-1, exp_name=exp_name)
  assert isinstance(model, LSTM2002)

  device = torch.device('cpu')
  for p in model.parameters():
    assert p.device == device, 'Must save model parameters to CPU.'

  # Must log training performance to 6 digits after decimal point.
  captured = capsys.readouterr()
  assert re.search(r'loss: \d+\.\d{6}', captured.err)
