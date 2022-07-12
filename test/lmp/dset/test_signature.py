"""Test :py:mod:`lmp.dset` signatures."""

import lmp.dset


def test_module_attribute() -> None:
  """Ensure module attributes' signatures."""
  assert hasattr(lmp.dset, 'BaseDset')
  assert hasattr(lmp.dset, 'ChPoemDset')
  assert hasattr(lmp.dset, 'DemoDset')
  assert hasattr(lmp.dset, 'WNLIDset')
  assert hasattr(lmp.dset, 'WikiText2Dset')
  assert hasattr(lmp.dset, 'ALL_DSETS')
  assert lmp.dset.ALL_DSETS == [
    lmp.dset.ChPoemDset,
    lmp.dset.DemoDset,
    lmp.dset.WNLIDset,
    lmp.dset.WikiText2Dset,
  ]
  assert hasattr(lmp.dset, 'DSET_OPTS')
  assert lmp.dset.DSET_OPTS == {
    lmp.dset.ChPoemDset.dset_name: lmp.dset.ChPoemDset,
    lmp.dset.DemoDset.dset_name: lmp.dset.DemoDset,
    lmp.dset.WNLIDset.dset_name: lmp.dset.WNLIDset,
    lmp.dset.WikiText2Dset.dset_name: lmp.dset.WikiText2Dset,
  }
