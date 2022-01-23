"""Test parsing arguments.

Test target:
- :py:meth:`lmp.script.gen_txt.parse_args`.
"""

import lmp.infer
import lmp.script.gen_txt
from lmp.infer import Top1Infer, TopKInfer, TopPInfer


def test_top_1_parse_results(ckpt: int, exp_name: str, max_seq_len: int, seed: int) -> None:
  """Must correctly parse all arguments for :py:class:`lmp.infer.Top1Infer`."""
  txt = 'Hello world'
  args = lmp.script.gen_txt.parse_args(
    argv=[
      Top1Infer.infer_name,
      '--ckpt',
      str(ckpt),
      '--exp_name',
      exp_name,
      '--max_seq_len',
      str(max_seq_len),
      '--seed',
      str(seed),
      '--txt',
      txt,
    ]
  )
  assert args.ckpt == ckpt
  assert args.exp_name == exp_name
  assert args.infer_name == Top1Infer.infer_name
  assert args.max_seq_len == max_seq_len
  assert args.seed == seed
  assert args.txt == txt


def test_top_k_parse_results(ckpt: int, exp_name: str, max_seq_len: int, seed: int) -> None:
  """Must correctly parse all arguments for :py:class:`lmp.infer.TopKInfer`."""
  k = 5
  txt = 'Hello world'
  args = lmp.script.gen_txt.parse_args(
    argv=[
      TopKInfer.infer_name,
      '--ckpt',
      str(ckpt),
      '--exp_name',
      exp_name,
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
  assert args.ckpt == ckpt
  assert args.exp_name == exp_name
  assert args.infer_name == TopKInfer.infer_name
  assert args.k == k
  assert args.max_seq_len == max_seq_len
  assert args.seed == seed
  assert args.txt == txt


def test_top_p_parse_results(ckpt: int, exp_name: str, max_seq_len: int, seed: int) -> None:
  """Must correctly parse all arguments for :py:class:`lmp.infer.TopPInfer`."""
  p = 0.9
  txt = 'Hello world'
  args = lmp.script.gen_txt.parse_args(
    argv=[
      TopPInfer.infer_name,
      '--ckpt',
      str(ckpt),
      '--exp_name',
      exp_name,
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
  assert args.ckpt == ckpt
  assert args.exp_name == exp_name
  assert args.infer_name == TopPInfer.infer_name
  assert args.max_seq_len == max_seq_len
  assert args.p == p
  assert args.seed == seed
  assert args.txt == txt
