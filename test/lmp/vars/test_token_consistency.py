"""Test token consistency.

Test target:
- :py:attr:`lmp.vars.BOS_TK`.
- :py:attr:`lmp.vars.BOS_TKID`.
- :py:attr:`lmp.vars.EOS_TK`.
- :py:attr:`lmp.vars.EOS_TKID`.
- :py:attr:`lmp.vars.PAD_TK`.
- :py:attr:`lmp.vars.PAD_TKID`.
- :py:attr:`lmp.vars.UNK_TK`.
- :py:attr:`lmp.vars.UNK_TKID`.
"""

import lmp.vars


def test_token_consistency():
  """Tokens are the same across different versions."""
  assert lmp.vars.BOS_TK == '<bos>'
  assert lmp.vars.EOS_TK == '<eos>'
  assert lmp.vars.PAD_TK == '<pad>'
  assert lmp.vars.UNK_TK == '<unk>'


def test_token_id_consistency():
  """Token ids are the same across different versions."""
  assert lmp.vars.BOS_TKID == 0
  assert lmp.vars.EOS_TKID == 1
  assert lmp.vars.PAD_TKID == 2
  assert lmp.vars.UNK_TKID == 3
