"""Test training :py:class:`lmp.model.ElmanNet`.

Test target:
- :py:meth:`lmp.model.ElmanNet.forward`
- :py:meth:`lmp.script.train_model.main`.
"""

import math
import os
import re

import torch

import lmp.script.train_model
import lmp.util.cfg
import lmp.util.model
import lmp.util.path
from lmp.dset import DemoDset
from lmp.model import ElmanNet


def test_train_elman_net_on_demo(
  batch_size: int,
  beta1: float,
  beta2: float,
  capsys,
  cfg_file_path: str,
  ckpt_dir_path: str,
  ckpt_step: int,
  ctx_win: int,
  d_emb: int,
  d_hid: int,
  eps: float,
  exp_name: str,
  log_step: int,
  lr: float,
  max_norm: float,
  max_seq_len: int,
  n_lyr: int,
  p_emb: float,
  p_hid: float,
  seed: int,
  tknzr_exp_name: str,
  total_step: int,
  train_log_dir_path: str,
  warmup_step: int,
  wd: float,
) -> None:
  """Successfully train model :py:class:`lmp.model.ElmanNet` on :py:class:`lmp.dset.DemoDset` dataset."""
  argv = [
    ElmanNet.model_name,
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
    '--d_emb',
    str(d_emb),
    '--d_hid',
    str(d_hid),
    '--dset_name',
    DemoDset.dset_name,
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
    '--n_lyr',
    str(n_lyr),
    '--p_emb',
    str(p_emb),
    '--p_hid',
    str(p_hid),
    '--seed',
    str(seed),
    '--tknzr_exp_name',
    str(tknzr_exp_name),
    '--total_step',
    str(total_step),
    '--ver',
    'valid',  # Make training faster.
    '--warmup_step',
    str(warmup_step),
    '--wd',
    str(wd),
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
  assert cfg.ctx_win == ctx_win
  assert cfg.d_emb == d_emb
  assert cfg.d_hid == d_hid
  assert cfg.dset_name == DemoDset.dset_name
  assert math.isclose(cfg.eps, eps)
  assert cfg.exp_name == exp_name
  assert cfg.log_step == log_step
  assert math.isclose(cfg.lr, lr)
  assert math.isclose(cfg.max_norm, max_norm)
  assert cfg.max_seq_len == max_seq_len
  assert cfg.model_name == ElmanNet.model_name
  assert math.isclose(cfg.p_emb, p_emb)
  assert math.isclose(cfg.p_hid, p_hid)
  assert cfg.seed == seed
  assert cfg.tknzr_exp_name == tknzr_exp_name
  assert cfg.total_step == total_step
  assert cfg.ver == 'valid'
  assert cfg.warmup_step == warmup_step
  assert math.isclose(cfg.wd, wd)

  model = lmp.util.model.load(ckpt=-1, exp_name=exp_name)
  assert isinstance(model, ElmanNet)

  device = torch.device('cpu')
  for p in model.parameters():
    assert p.device == device, 'Must save model parameters to CPU.'

  # Must log training performance to 6 digits after decimal point.
  captured = capsys.readouterr()
  assert re.search(r'loss: \d+\.\d{6}', captured.err)
