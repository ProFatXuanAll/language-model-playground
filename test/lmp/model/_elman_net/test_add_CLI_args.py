"""Test parser arguments.

Test target:
- :py:meth:`lmp.model._elman_net.ElmanNet.add_CLI_args`.
"""

import argparse
import math

from lmp.model._elman_net import ElmanNet


def test_default_value() -> None:
  """Ensure default value consistency."""
  d_emb = 1
  d_hid = 1
  init_lower = -0.1
  init_upper = 0.1
  label_smoothing = 0.0
  n_lyr = 1
  p_emb = 0.0
  p_hid = 0.0
  parser = argparse.ArgumentParser()
  ElmanNet.add_CLI_args(parser=parser)
  args = parser.parse_args([])

  assert args.d_emb == d_emb
  assert args.d_hid == d_hid
  assert math.isclose(args.init_lower, init_lower)
  assert math.isclose(args.init_upper, init_upper)
  assert math.isclose(args.label_smoothing, label_smoothing)
  assert args.n_lyr == n_lyr
  assert math.isclose(args.p_emb, p_emb)
  assert math.isclose(args.p_hid, p_hid)


def test_arguments(
  d_emb: int,
  d_hid: int,
  init_lower: float,
  init_upper: float,
  label_smoothing: float,
  n_lyr: int,
  p_emb: float,
  p_hid: float,
) -> None:
  """Must have correct arguments."""
  parser = argparse.ArgumentParser()
  ElmanNet.add_CLI_args(parser=parser)
  args = parser.parse_args(
    [
      '--d_emb',
      str(d_emb),
      '--d_hid',
      str(d_hid),
      '--init_lower',
      str(init_lower),
      '--init_upper',
      str(init_upper),
      '--label_smoothing',
      str(label_smoothing),
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
  assert math.isclose(args.init_lower, init_lower)
  assert math.isclose(args.init_upper, init_upper)
  assert math.isclose(args.label_smoothing, label_smoothing)
  assert args.n_lyr == n_lyr
  assert math.isclose(args.p_emb, p_emb)
  assert math.isclose(args.p_hid, p_hid)
