"""Test training :py:class:`lmp.model.LSTM2000`.

Test target:
- :py:meth:`lmp.model.LSTM2000.forward`
- :py:meth:`lmp.script.ddp_train_model.main`.
"""

import math
import multiprocessing as mp
import os

import torch

import lmp.script.ddp_train_model
import lmp.util.cfg
import lmp.util.model
import lmp.util.path
from lmp.dset import WikiText2Dset
from lmp.model import LSTM2000


def test_train_elman_net_on_wiki_text_2(
  batch_size: int,
  beta1: float,
  beta2: float,
  cfg_file_path: str,
  ckpt_dir_path: str,
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
  train_log_dir_path: str,
  warmup_step: int,
  wd: float,
  world_size: int,
) -> None:
  """Successfully train model :py:class:`lmp.model.LSTM2000` on :py:class:`lmp.dset.WikiText2Dset` dataset."""
  # This is need since one cannot re-initialize CUDA in forked subprocess.
  ctx = mp.get_context('spawn')

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
    WikiText2Dset.dset_name,
    '--eps',
    str(eps),
    '--exp_name',
    exp_name,
    '--host_name',
    host_name,
    '--host_port',
    str(host_port),
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
    'valid',  # Make training faster.
    '--warmup_step',
    str(warmup_step),
    '--wd',
    str(wd),
    '--world_size',
    str(world_size),
  ]

  if is_dset_in_memory:
    argv.append('--is_dset_in_memory')

  if world_size == 1:
    argv.append('--rank')
    argv.append('0')
    argv.append('--local_rank')
    argv.append('0')
    lmp.script.ddp_train_model.main(argv=argv)
  else:
    p0 = ctx.Process(target=lmp.script.ddp_train_model.main, args=(argv + ['--rank', '0', '--local_rank', '0'],))
    p1 = ctx.Process(target=lmp.script.ddp_train_model.main, args=(argv + ['--rank', '1', '--local_rank', '1'],))
    p0.start()
    p1.start()
    p0.join()
    p1.join()
    assert p0.exitcode == p1.exitcode == 0
    p0.close()
    p1.close()

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
  assert cfg.dset_name == WikiText2Dset.dset_name
  assert math.isclose(cfg.eps, eps)
  assert cfg.exp_name == exp_name
  assert cfg.is_dset_in_memory == is_dset_in_memory
  assert cfg.local_rank == 0
  assert cfg.log_step == log_step
  assert math.isclose(cfg.lr, lr)
  assert math.isclose(cfg.max_norm, max_norm)
  assert cfg.max_seq_len == max_seq_len
  assert cfg.model_name == LSTM2000.model_name
  assert cfg.n_blk == n_blk
  assert cfg.n_epoch == n_epoch
  assert cfg.n_worker == n_worker
  assert math.isclose(cfg.p_emb, p_emb)
  assert math.isclose(cfg.p_hid, p_hid)
  assert cfg.rank == 0
  assert cfg.seed == seed
  assert cfg.tknzr_exp_name == tknzr_exp_name
  assert cfg.ver == 'valid'
  assert cfg.warmup_step == warmup_step
  assert math.isclose(cfg.wd, wd)
  assert cfg.world_size == world_size

  model = lmp.util.model.load(ckpt=-1, exp_name=exp_name)
  assert isinstance(model, LSTM2000)

  device = torch.device('cpu')
  for p in model.parameters():
    assert p.device == device, 'Must save model parameters to CPU.'
