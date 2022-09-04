"""Test parser arguments.

Test target:
- :py:meth:`lmp.model._lstm_1997.LSTM1997.add_CLI_args`.
"""

import argparse
import math

from lmp.model._lstm_1997 import LSTM1997


def test_default_value() -> None:
  """Ensure default value consistency."""
  d_blk = 1
  d_emb = 1
  init_ib = -1.0
  init_lower = -0.1
  init_ob = -1.0
  init_upper = 0.1
  label_smoothing = 0.0
  n_blk = 1
  n_lyr = 1
  p_emb = 0.0
  p_hid = 0.0
  parser = argparse.ArgumentParser()
  LSTM1997.add_CLI_args(parser=parser)
  args = parser.parse_args([])

  assert args.d_blk == d_blk
  assert args.d_emb == d_emb
  assert math.isclose(args.init_ib, init_ib)
  assert math.isclose(args.init_lower, init_lower)
  assert math.isclose(args.init_ob, init_ob)
  assert math.isclose(args.init_upper, init_upper)
  assert math.isclose(args.label_smoothing, label_smoothing)
  assert args.n_blk == n_blk
  assert args.n_lyr == n_lyr
  assert math.isclose(args.p_emb, p_emb)
  assert math.isclose(args.p_hid, p_hid)


def test_arguments(
  d_blk: int,
  d_emb: int,
  init_ib: float,
  init_lower: float,
  init_ob: float,
  init_upper: float,
  label_smoothing: float,
  n_blk: int,
  n_lyr: int,
  p_emb: float,
  p_hid: float,
) -> None:
  """Must have correct arguments."""
  parser = argparse.ArgumentParser()
  LSTM1997.add_CLI_args(parser=parser)
  args = parser.parse_args(
    [
      '--d_blk',
      str(d_blk),
      '--d_emb',
      str(d_emb),
      '--init_ib',
      str(init_ib),
      '--init_lower',
      str(init_lower),
      '--init_ob',
      str(init_ob),
      '--init_upper',
      str(init_upper),
      '--label_smoothing',
      str(label_smoothing),
      '--n_blk',
      str(n_blk),
      '--n_lyr',
      str(n_lyr),
      '--p_emb',
      str(p_emb),
      '--p_hid',
      str(p_hid),
    ]
  )

  assert args.d_blk == d_blk
  assert args.d_emb == d_emb
  assert math.isclose(args.init_ib, init_ib)
  assert math.isclose(args.init_lower, init_lower)
  assert math.isclose(args.init_ob, init_ob)
  assert math.isclose(args.init_upper, init_upper)
  assert math.isclose(args.label_smoothing, label_smoothing)
  assert args.n_blk == n_blk
  assert args.n_lyr == n_lyr
  assert math.isclose(args.p_emb, p_emb)
  assert math.isclose(args.p_hid, p_hid)
