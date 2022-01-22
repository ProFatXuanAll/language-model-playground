"""Test random seed setting utilities.

Test target:
- :py:meth:`lmp.util.rand.set_seed`.
"""

import random

import lmp.util.rand


def test_set_seed():
  """Setting random seeds."""
  lmp.util.rand.set_seed(seed=1)
  state1 = random.getstate()
  lmp.util.rand.set_seed(seed=1)
  state2 = random.getstate()
  assert state1 == state2
