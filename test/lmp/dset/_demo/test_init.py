"""Test the construction of :py:class:`lmp.dset._demo.DemoDset`.

Test target:
- :py:meth:`lmp.dset._demo.DemoDset.__init__`.
"""

import re

import lmp.dset._demo


def test_default_version() -> None:
  """Must be able to construct the default version."""
  dset = lmp.dset._demo.DemoDset(ver=None)
  assert dset.ver == lmp.dset._demo.DemoDset.df_ver


def test_all_verions() -> None:
  """Must be able to construct all versions of :py:class:`lmp.dset._demo.DemoDset`."""
  for ver in lmp.dset._demo.DemoDset.vers:
    dset = lmp.dset._demo.DemoDset(ver=ver)
    assert dset.ver == ver
    assert len(dset) > 0
    assert all(map(lambda spl: isinstance(spl, str), dset))
    assert all(map(lambda spl: len(spl) > 0, dset))


def test_consistent_format() -> None:
  """Must have consistent format."""
  pttn = re.compile(r'If you add (\d+) to (\d+) you get (\d+) \.')
  for ver in lmp.dset._demo.DemoDset.vers:
    for spl in lmp.dset._demo.DemoDset(ver=ver):
      match = pttn.match(spl)
      assert match

      num_1 = int(match.group(1))
      num_2 = int(match.group(2))
      assert 0 <= num_1 <= 99
      assert 0 <= num_2 <= 99
      assert num_1 + num_2 == int(match.group(3))


def test_mutually_exclusive() -> None:
  """Different versions are mutually exclusive."""
  dsets = []
  total_size = 0
  for ver in lmp.dset._demo.DemoDset.vers:
    dset = lmp.dset._demo.DemoDset(ver=ver)
    total_size += len(dset)
    dsets.append(set(dset))

  # Check mutually exclusive.  If different versions of dataset are mutually exclusive, then their union size must be
  # the total number of dataset samples.
  dset_union = set()
  for dset in dsets:
    dset_union = dset_union | dset
  assert len(dset_union) == total_size


def test_commutative() -> None:
  """Training and validation sets are consist of commutative pairs of additions."""
  pttn = re.compile(r'If you add (\d+) to (\d+) you get \d+ \.')
  train = lmp.dset._demo.DemoDset(ver='train')
  valid = lmp.dset._demo.DemoDset(ver='valid')

  train_pool = set()
  for spl in train:
    match = pttn.match(spl)
    num_1 = match.group(1)
    num_2 = match.group(2)
    train_pool.add((num_1, num_2))

  # If `a + b` is in training set, then `b + a` must be in validation set.
  for spl in valid:
    match = pttn.match(spl)
    num_1 = match.group(1)
    num_2 = match.group(2)
    assert (num_2, num_1) in train_pool
