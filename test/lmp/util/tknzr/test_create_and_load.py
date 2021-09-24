r"""Test operation for new tokenizer.

Test target:
- :py:meth:`lmp.util.tknzr.create`.
- :py:meth:`lmp.util.tknzr.load`.
"""

from lmp.tknzr import WsTknzr
import lmp.util.tknzr


def test_create(
    exp_name,
):
    r"""Ensure create the correct``whitespace`` tokenizer."""
    # Test Case: Creation tokenizer type check.
    tknzr = lmp.util.tknzr.create(
        is_uncased=False,
        max_vocab=-1,
        min_count=1,
        tknzr_name='whitespace',
    )

    isinstance(tknzr, WsTknzr)


def test_load(
    exp_name,
    clean_tknzr,
):
    r"""Ensure the consistency between creation and load."""
    tknzr = lmp.util.tknzr.create(
        is_uncased=False,
        max_vocab=-1,
        min_count=1,
        tknzr_name='whitespace',
    )
    tknzr.save(
        exp_name=exp_name
    )

    load_tknzr = lmp.util.tknzr.load(
        exp_name=exp_name,
        tknzr_name='whitespace'
    )

    # Test Case: Load tokenizer type check
    isinstance(load_tknzr, WsTknzr)

    # Test Case: Load tokenzier value check.
    assert not tknzr.is_uncased
    assert tknzr.max_vocab == -1
    assert tknzr.min_count == 1
