r"""Test the constructor of :py:class:`lmp.dset.WikiText2Dset`.

Test target:
- :py:meth:`lmp.dset.WikiText2Dset.__init__`.
"""

import pytest
from lmp.dset import WikiText2Dset


def test_ver():
    r"""``ver`` parameter must be included in :py:attr:`lmp.dset.WikiText2Dset.vers`."""

    # Test case: Type mismatched.
    wrong_typed_inputs = [
        False, True, 0, 1, -1, 0.1, (), [], {}, set(), ..., NotImplemented,
    ]

    for bad_ver in wrong_typed_inputs:
        with pytest.raises(TypeError) as excinfo:
            WikiText2Dset(ver=bad_ver)

        assert '`ver` must be an instance of `str`' in str(excinfo.value)

    # Test case: Unsupported versions.
    wrong_value_inputs = ['', '123']

    for bad_ver in wrong_value_inputs:
        with pytest.raises(ValueError) as excinfo:
            WikiText2Dset(ver=bad_ver)

        assert f'Version {bad_ver} is not available' in str(excinfo.value)
        assert 'Available versions' in str(excinfo.value)

        for good_ver in WikiText2Dset.vers:
            assert good_ver in str(excinfo.value)

    # Test case: Supported versions.
    for good_ver in WikiText2Dset.vers:
        dset = WikiText2Dset(ver=good_ver)
        assert dset.ver == good_ver

    # Test case: Default version.
    dset = WikiText2Dset(ver=None)
    assert dset.ver == WikiText2Dset.df_ver
