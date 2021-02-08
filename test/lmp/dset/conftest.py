r"""Setup fixture for testing :py:mod:`lmp.dset`."""

import pytest


@pytest.fixture(params=[])
def ver(request) -> str:
    r"""Dataset instance attribute ``ver``."""
    return request.param
