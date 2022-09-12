"""Test parser arguments.

Test target:
- :py:meth:`lmp.model._trans_enc.TransEnc.add_CLI_args`.
"""

import argparse
import math

from lmp.model._trans_enc import TransEnc


def test_default_value() -> None:
  """Ensure default value consistency."""
  d_ff = 1
  d_k = 1
  d_model = 1
  d_v = 1
  init_lower = -0.1
  init_upper = 0.1
  label_smoothing = 0.0
  n_head = 1
  n_lyr = 1
  p = 0.0
  parser = argparse.ArgumentParser()
  TransEnc.add_CLI_args(parser=parser)
  args = parser.parse_args([])

  assert args.d_ff == d_ff
  assert args.d_k == d_k
  assert args.d_model == d_model
  assert args.d_v == d_v
  assert math.isclose(args.init_lower, init_lower)
  assert math.isclose(args.init_upper, init_upper)
  assert math.isclose(args.label_smoothing, label_smoothing)
  assert args.n_head == n_head
  assert args.n_lyr == n_lyr
  assert math.isclose(args.p, p)


def test_arguments(
  d_ff: int,
  d_k: int,
  d_model: int,
  d_v: int,
  init_lower: float,
  init_upper: float,
  label_smoothing: float,
  n_head: int,
  n_lyr: int,
  p_hid: float,
) -> None:
  """Must have correct arguments."""
  parser = argparse.ArgumentParser()
  TransEnc.add_CLI_args(parser=parser)
  args = parser.parse_args(
    [
      '--d_ff',
      str(d_ff),
      '--d_k',
      str(d_k),
      '--d_model',
      str(d_model),
      '--d_v',
      str(d_v),
      '--init_lower',
      str(init_lower),
      '--init_upper',
      str(init_upper),
      '--label_smoothing',
      str(label_smoothing),
      '--n_head',
      str(n_head),
      '--n_lyr',
      str(n_lyr),
      '--p',
      str(p_hid),
    ]
  )

  assert args.d_ff == d_ff
  assert args.d_k == d_k
  assert args.d_model == d_model
  assert args.d_v == d_v
  assert math.isclose(args.init_lower, init_lower)
  assert math.isclose(args.init_upper, init_upper)
  assert math.isclose(args.label_smoothing, label_smoothing)
  assert args.n_head == n_head
  assert args.n_lyr == n_lyr
  assert math.isclose(args.p, p_hid)
