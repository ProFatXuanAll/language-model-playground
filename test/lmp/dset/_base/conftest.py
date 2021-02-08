r"""Setup fixture for testing :py:mod:`lmp.dset.BaseDset`."""

from typing import Type

import pytest

from lmp.dset._base import BaseDset


@pytest.fixture
def subclss_dset_clss() -> Type[BaseDset]:
    r"""Simple ``BaseDset`` subclass."""
    class SubclssDset(BaseDset):
        pass
    return SubclssDset


@pytest.fixture
def subclss_dset(
        subclss_dset_clss: Type[BaseDset],
        ver: str,
):
    r"""Simple ``BaseDset`` subclass instance."""
    return subclss_dset_clss(
        ver=ver,
    )
