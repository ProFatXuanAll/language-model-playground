"""Test parser arguments.

Test target:
- :py:meth:`lmp.model._lstm_2002.LSTM2002.add_CLI_args`.
"""

import argparse
import math

from lmp.model._lstm_2002 import LSTM2002


def test_arguments(d_blk: int, d_emb: int, n_blk: int, p_emb: float, p_hid: float) -> None:
  """Must have correct arguments."""
  parser = argparse.ArgumentParser()
  LSTM2002.add_CLI_args(parser=parser)
  args = parser.parse_args(
    [
      '--d_blk',
      str(d_blk),
      '--d_emb',
      str(d_emb),
      '--n_blk',
      str(n_blk),
      '--p_emb',
      str(p_emb),
      '--p_hid',
      str(p_hid),
    ]
  )
  assert args.d_blk == d_blk
  assert args.d_emb == d_emb
  assert args.n_blk == n_blk
  assert math.isclose(args.p_emb, p_emb)
  assert math.isclose(args.p_hid, p_hid)
