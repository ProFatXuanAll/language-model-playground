r"""Test sampling from dataset.

Test target:
- :py:meth:`lmp.dset.ChPoemDset.__getitem__`.
- :py:meth:`lmp.dset.ChPoemDset.__iter__`.
- :py:meth:`lmp.dset.ChPoemDset.__len__`.
"""

from lmp.dset import ChPoemDset


def test_dataset_size():
    r"""Dataset size must be larger than ``0``."""

    for ver in ChPoemDset.vers:
        dset = ChPoemDset(ver=ver)
        assert len(dset) > 0


def test_dataset_sample():
    r"""Sample in dataset must be instances of ``str``."""

    for ver in ChPoemDset.vers:
        dset = ChPoemDset(ver=ver)

        for idx in range(len(dset)):
            assert isinstance(dset[idx], str)


def test_sampling_order():
    r"""Sampling order must always be the same."""

    for ver in ChPoemDset.vers:
        dset = ChPoemDset(ver=ver)

        order_1 = iter(dset)
        order_2 = iter(dset)

        for sample_1, sample_2 in zip(order_1, order_2):
            assert sample_1 == sample_2
