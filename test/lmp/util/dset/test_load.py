r"""Test loading the dataset.

Test target:
- :py:meth:`lmp.util.dset.load`.
"""

from lmp.dset import WikiText2Dset
from lmp.util.dset import load


def test_load(
    clean_dset,
) -> WikiText2Dset:
    r"""Ensure load the correct ``wikitext2`` dataset."""
    dset = load(
        dset_name='wikitext-2',
        ver='valid',
    )

    assert isinstance(dset, WikiText2Dset)

    assert dset.ver == 'valid'
    assert dset.dset_name == 'wikitext-2'
