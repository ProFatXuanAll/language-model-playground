"""Test sampling from dataset.

Test target:
- :py:meth:`lmp.dset._wiki_text_2.WikiText2Dset.__getitem__`.
- :py:meth:`lmp.dset._wiki_text_2.WikiText2Dset.__iter__`.
- :py:meth:`lmp.dset._wiki_text_2.WikiText2Dset.__len__`.
"""

import pytest

import lmp.dset._wiki_text_2


def test_sampling_order() -> None:
  """Sample order must always be the same."""
  for ver in lmp.dset._wiki_text_2.WikiText2Dset.vers:
    dset = lmp.dset._wiki_text_2.WikiText2Dset(ver=ver)

    order_1 = iter(dset)
    order_2 = iter(dset)

    # Python3.8 does not have `strict` argument in `zip()` function.
    for idx, spl in enumerate(dset):
      assert spl == dset[idx] == next(order_1) == next(order_2)

    # Must have no more samples.
    with pytest.raises(StopIteration):
      next(order_1)

    # Must have no more samples.
    with pytest.raises(StopIteration):
      next(order_2)

    # Must have correct number of samples.
    assert idx + 1 == len(dset)
