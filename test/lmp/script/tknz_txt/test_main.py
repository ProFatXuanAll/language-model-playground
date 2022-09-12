"""Test tokenization result.

Test target:
- :py:meth:`lmp.script.tknz_txt.main`.
"""

import lmp.script.tknz_txt
import lmp.util.cfg
import lmp.util.tknzr
from lmp.tknzr import BPETknzr, CharTknzr, WsTknzr


def test_bpe_tknzr(capsys, bpe_tknzr: BPETknzr, exp_name: str, seed: int) -> None:
  """Ensure tokenize script output consistency when using :py:class:`lmp.tknzr.BPETknzr`."""
  txt = 'abc'

  assert lmp.script.tknz_txt.main(argv=[
    '--exp_name',
    exp_name,
    '--seed',
    str(seed),
    '--txt',
    txt,
  ]) == bpe_tknzr.tknz(txt=txt)


def test_char_tknzr(capsys, char_tknzr: CharTknzr, exp_name: str, seed: int) -> None:
  """Ensure tokenize script output consistency when using :py:class:`lmp.tknzr.CharTknzr`."""
  txt = 'abc'

  assert lmp.script.tknz_txt.main(argv=[
    '--exp_name',
    exp_name,
    '--seed',
    str(seed),
    '--txt',
    txt,
  ]) == char_tknzr.tknz(txt=txt)


def test_ws_tknzr(capsys, ws_tknzr: WsTknzr, exp_name: str, seed: int) -> None:
  """Ensure tokenize script output consistency when using :py:class:`lmp.tknzr.WsTknzr`."""
  txt = 'a b c'

  assert lmp.script.tknz_txt.main(argv=[
    '--exp_name',
    exp_name,
    '--seed',
    str(seed),
    '--txt',
    txt,
  ]) == ws_tknzr.tknz(txt=txt)
