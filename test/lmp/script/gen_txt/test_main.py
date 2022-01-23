"""Test generation results.

Test target:
- :py:meth:`lmp.script.gen_txt.main`.
"""

import lmp.script.gen_txt
from lmp.infer import Top1Infer, TopKInfer, TopPInfer
from lmp.tknzr import BaseTknzr


def test_top_1_output(capsys, ckpt: int, max_seq_len: int, model_exp_name: str, seed: int, tknzr: BaseTknzr) -> None:
  """Must correctly output generation result using :py:class:`lmp.infer.Top1Infer`."""
  txt = 'Hello world'
  lmp.script.gen_txt.main(
    argv=[
      Top1Infer.infer_name,
      '--ckpt',
      str(ckpt),
      '--exp_name',
      model_exp_name,
      '--max_seq_len',
      str(max_seq_len),
      '--seed',
      str(seed),
      '--txt',
      txt,
    ]
  )

  captured = capsys.readouterr()
  assert captured.out
  assert len(tknzr.tknz(captured.out)) <= max_seq_len


def test_top_k_output(capsys, ckpt: int, max_seq_len: int, model_exp_name: str, seed: int, tknzr: BaseTknzr) -> None:
  """Must correctly output generation result using :py:class:`lmp.infer.TopKInfer`."""
  k = 5
  txt = 'Hello world'
  lmp.script.gen_txt.main(
    argv=[
      TopKInfer.infer_name,
      '--ckpt',
      str(ckpt),
      '--exp_name',
      model_exp_name,
      '--k',
      str(k),
      '--max_seq_len',
      str(max_seq_len),
      '--seed',
      str(seed),
      '--txt',
      txt,
    ]
  )

  captured = capsys.readouterr()
  assert captured.out
  assert len(tknzr.tknz(captured.out)) <= max_seq_len


def test_top_p_output(capsys, ckpt: int, max_seq_len: int, model_exp_name: str, seed: int, tknzr: BaseTknzr) -> None:
  """Must correctly output generation result using :py:class:`lmp.infer.TopPInfer`."""
  p = 0.9
  txt = 'Hello world'
  lmp.script.gen_txt.main(
    argv=[
      TopPInfer.infer_name,
      '--ckpt',
      str(ckpt),
      '--exp_name',
      model_exp_name,
      '--max_seq_len',
      str(max_seq_len),
      '--p',
      str(p),
      '--seed',
      str(seed),
      '--txt',
      txt,
    ]
  )

  captured = capsys.readouterr()
  assert captured.out
  assert len(tknzr.tknz(captured.out)) <= max_seq_len
