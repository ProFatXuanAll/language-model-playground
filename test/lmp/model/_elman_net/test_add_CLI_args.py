"""Test parser arguments.

Test target:
- :py:meth:`lmp.model._elman_net.ElmanNet.add_CLI_args`.
"""

import argparse
import math

from lmp.model._elman_net import ElmanNet


def test_arguments(d_emb: int, d_hid: int, n_lyr: int, p_emb: float, p_hid: float) -> None:
  """Must have correct arguments."""
  parser = argparse.ArgumentParser()
  ElmanNet.add_CLI_args(parser=parser)
  args = parser.parse_args(
    [
      '--d_emb',
      str(d_emb),
      '--d_hid',
      str(d_hid),
      '--n_lyr',
      str(n_lyr),
      '--p_emb',
      str(p_emb),
      '--p_hid',
      str(p_hid),
    ]
  )
  assert args.d_emb == d_emb
  assert args.d_hid == d_hid
  assert args.n_lyr == n_lyr
  assert math.isclose(args.p_emb, p_emb)
  assert math.isclose(args.p_hid, p_hid)
