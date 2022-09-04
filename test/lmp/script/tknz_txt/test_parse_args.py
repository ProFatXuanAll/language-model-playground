"""Test parsing arguments.

Test target:
- :py:meth:`lmp.script.tknz_txt.parse_args`.
"""

import lmp.script.tknz_txt


def test_default_values() -> None:
  """Ensure default values consistency."""
  args = lmp.script.tknz_txt.parse_args(argv=[])
  assert args.exp_name == 'my_tknzr_exp'
  assert args.seed == 42
  assert args.txt == ''


def test_parse_results(exp_name: str, seed: int) -> None:
  """Must correctly parse all arguments."""
  txt = 'abc'
  args = lmp.script.tknz_txt.parse_args(argv=[
    '--exp_name',
    exp_name,
    '--seed',
    str(seed),
    '--txt',
    txt,
  ])
  assert args.exp_name == exp_name
  assert args.seed == seed
  assert args.txt == txt
