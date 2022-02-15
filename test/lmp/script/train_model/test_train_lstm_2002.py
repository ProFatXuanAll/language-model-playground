"""Test training :py:class:`lmp.model.LSTM2002`.

Test target:
- :py:meth:`lmp.script.train_tknzr.main`.
"""

import os

import lmp.script.train_tknzr
import lmp.util.cfg
import lmp.util.model
import lmp.util.path
from lmp.dset import WikiText2Dset
from lmp.model import LSTM2002


def test_train_lstm_2002_on_wiki_text_2(
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
  log_dir_path: str,
  log_step: int,
  lr: float,
  max_norm: float,
  max_seq_len: int,
  n_blk: int,
  n_epoch: int,
  seed: int,
  tknzr_exp_name: str,
  wd: float,
) -> None:
  """Successfully train model :py:class:`lmp.model.LSTM2002` on :py:class:`lmp.dset.WikiText2Dset` dataset."""
  lmp.script.train_model.main(
    argv=[
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
      WikiText2Dset.dset_name,
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
      '--seed',
      str(seed),
      '--tknzr_exp_name',
      str(tknzr_exp_name),
      '--ver',
      'valid',  # avoid training too long.
      '--wd',
      str(wd),
    ]
  )

  # Must save training configuration.
  assert os.path.exists(cfg_file_path)
  # Must save model checkpoints.
  assert os.path.exists(ckpt_dir_path)
  # Must have at least one checkpoints.
  assert os.path.exists(os.path.join(ckpt_dir_path, f'model-{ckpt_step}.pt'))
  # Must log model performance.
  assert os.path.exists(log_dir_path)

  cfg = lmp.util.cfg.load(exp_name=exp_name)
  assert cfg.batch_size == batch_size
  assert cfg.beta1 == beta1
  assert cfg.beta2 == beta2
  assert cfg.ckpt_step == ckpt_step
  assert cfg.d_blk == d_blk
  assert cfg.d_emb == d_emb
  assert cfg.dset_name == WikiText2Dset.dset_name
  assert cfg.eps == eps
  assert cfg.exp_name == exp_name
  assert cfg.log_step == log_step
  assert cfg.lr == lr
  assert cfg.max_norm == max_norm
  assert cfg.max_seq_len == max_seq_len
  assert cfg.model_name == LSTM2002.model_name
  assert cfg.n_blk == n_blk
  assert cfg.n_epoch == n_epoch
  assert cfg.seed == seed
  assert cfg.tknzr_exp_name == tknzr_exp_name
  assert cfg.ver == 'valid'
  assert cfg.wd == wd

  model = lmp.util.model.load(ckpt=-1, exp_name=exp_name)
  assert isinstance(model, LSTM2002)

  # Must log training performance.
  captured = capsys.readouterr()
  assert 'epoch' in captured.err
  assert 'loss' in captured.err
