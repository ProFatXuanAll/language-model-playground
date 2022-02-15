"""Test :py:mod:`lmp.tknzr` signatures."""

import lmp.tknzr


def test_module_attribute() -> None:
  """Ensure module attributes' signatures."""
  assert hasattr(lmp.tknzr, 'BaseTknzr')
  assert hasattr(lmp.tknzr, 'CharTknzr')
  assert hasattr(lmp.tknzr, 'WsTknzr')
  assert hasattr(lmp.tknzr, 'ALL_TKNZR')
  assert lmp.tknzr.ALL_TKNZRS == [
    lmp.tknzr.CharTknzr,
    lmp.tknzr.WsTknzr,
  ]
  assert hasattr(lmp.tknzr, 'TKNZR_OPTS')
  assert lmp.tknzr.TKNZR_OPTS == {
    lmp.tknzr.CharTknzr.tknzr_name: lmp.tknzr.CharTknzr,
    lmp.tknzr.WsTknzr.tknzr_name: lmp.tknzr.WsTknzr,
  }
