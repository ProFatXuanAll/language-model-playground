"""Test parsing arguments.

Test target:
- :py:meth:`lmp.script.tknz_txt.parse_args`.
"""

import lmp.script.tknz_txt


def test_parse_results(exp_name: str) -> None:
  """Must correctly parse all arguments."""
  txt = 'abc'
  args = lmp.script.tknz_txt.parse_args(argv=[
    '--exp_name',
    exp_name,
    '--txt',
    txt,
  ])
  assert args.exp_name == exp_name
  assert args.txt == txt
