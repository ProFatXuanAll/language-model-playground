"""Test parsing arguments for top-1 inference method.

Test target:
- :py:meth:`lmp.script.gen_txt.parse_args`.
"""

import lmp.infer
import lmp.script.gen_txt
from lmp.infer import Top1Infer


def test_default_value() -> None:
  """Ensure default value consistency."""
  ckpt = -1
  exp_name = 'my_model_exp'
  infer_name = Top1Infer.infer_name
  max_seq_len = 32
  seed = 42
  txt = ''

  args = lmp.script.gen_txt.parse_args(argv=[infer_name])

  assert args.ckpt == ckpt
  assert args.exp_name == exp_name
  assert args.infer_name == infer_name
  assert args.max_seq_len == max_seq_len
  assert args.seed == seed
  assert args.txt == txt


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
