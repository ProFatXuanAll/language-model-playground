"""Test parsing arguments for top-p inference method.

Test target:
- :py:meth:`lmp.script.gen_txt.parse_args`.
"""

import math

import lmp.infer
import lmp.script.gen_txt
from lmp.infer import TopPInfer


def test_default_value() -> None:
  """Ensure default value consistency."""
  ckpt = -1
  exp_name = 'my_model_exp'
  infer_name = TopPInfer.infer_name
  max_seq_len = 32
  p = 0.9
  seed = 42
  txt = ''

  args = lmp.script.gen_txt.parse_args(argv=[infer_name])

  assert args.ckpt == ckpt
  assert args.exp_name == exp_name
  assert args.infer_name == infer_name
  assert args.max_seq_len == max_seq_len
  assert math.isclose(args.p, p)
  assert args.seed == seed
  assert args.txt == txt


def test_top_p_parse_results(ckpt: int, exp_name: str, max_seq_len: int, p: float, seed: int) -> None:
  """Must correctly parse all arguments for :py:class:`lmp.infer.TopPInfer`."""
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
  assert math.isclose(args.p, p)
  assert args.seed == seed
  assert args.txt == txt
