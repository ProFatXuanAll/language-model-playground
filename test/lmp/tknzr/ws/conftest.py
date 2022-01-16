"""Setup fixtures for testing :py:class:`lmp.tknzr.WsTknzr`."""

from typing import Dict

import pytest

from lmp.tknzr import WsTknzr


@pytest.fixture(
  params=[
    {
      WsTknzr.bos_tk: WsTknzr.bos_tkid,
      WsTknzr.eos_tk: WsTknzr.eos_tkid,
      WsTknzr.pad_tk: WsTknzr.pad_tkid,
      WsTknzr.unk_tk: WsTknzr.unk_tkid,
    },
    {
      WsTknzr.bos_tk: WsTknzr.bos_tkid,
      WsTknzr.eos_tk: WsTknzr.eos_tkid,
      WsTknzr.pad_tk: WsTknzr.pad_tkid,
      WsTknzr.unk_tk: WsTknzr.unk_tkid,
      'a': max(WsTknzr.bos_tkid, WsTknzr.eos_tkid, WsTknzr.pad_tkid, WsTknzr.unk_tkid) + 1,
      'b': max(WsTknzr.bos_tkid, WsTknzr.eos_tkid, WsTknzr.pad_tkid, WsTknzr.unk_tkid) + 2,
      'c': max(WsTknzr.bos_tkid, WsTknzr.eos_tkid, WsTknzr.pad_tkid, WsTknzr.unk_tkid) + 3,
    },
  ]
)
def tk2id(request) -> Dict[str, int]:
  """Token-to-id lookup table."""
  return request.param
