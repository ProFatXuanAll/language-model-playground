"""Test parsing arguments for top-k inference method.

Test target:
- :py:meth:`lmp.script.gen_txt.parse_args`.
"""

import lmp.infer
import lmp.script.gen_txt
from lmp.infer import TopKInfer


def test_default_value() -> None:
  """Ensure default value consistency."""
  ckpt = -1
  exp_name = 'my_model_exp'
  infer_name = TopKInfer.infer_name
  k = 5
  max_seq_len = 32
  seed = 42
  txt = ''

  args = lmp.script.gen_txt.parse_args(argv=[infer_name])

  assert args.ckpt == ckpt
  assert args.exp_name == exp_name
  assert args.infer_name == infer_name
  assert args.k == k
  assert args.max_seq_len == max_seq_len
  assert args.seed == seed
  assert args.txt == txt


def test_top_k_parse_results(ckpt: int, exp_name: str, k: int, max_seq_len: int, seed: int) -> None:
  """Must correctly parse all arguments for :py:class:`lmp.infer.TopKInfer`."""
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
