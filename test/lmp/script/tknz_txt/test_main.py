"""Test tokenization result.

Test target:
- :py:meth:`lmp.script.tknz_txt.main`.
"""

import lmp.script.tknz_txt
import lmp.util.cfg
import lmp.util.tknzr
from lmp.tknzr import CharTknzr, WsTknzr


def test_char_tknzr(capsys, char_tknzr: CharTknzr, exp_name: str) -> None:
  """Ensure tokenize script output consistency when using :py:class:`lmp.tknzr.CharTknzr`."""
  txt = 'abc'

  lmp.script.tknz_txt.main(argv=[
    '--exp_name',
    exp_name,
    '--txt',
    txt,
  ])

  captured = capsys.readouterr()
  assert str(char_tknzr.tknz(txt=txt)) in captured.out


def test_ws_tknzr(capsys, ws_tknzr: WsTknzr, exp_name: str) -> None:
  """Ensure tokenize script output consistency when using :py:class:`lmp.tknzr.WsTknzr`."""
  txt = 'a b c'

  lmp.script.tknz_txt.main(argv=[
    '--exp_name',
    exp_name,
    '--txt',
    txt,
  ])

  captured = capsys.readouterr()
  assert str(ws_tknzr.tknz(txt=txt)) in captured.out
