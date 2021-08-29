r"""Test the construction of :py:mod:`lmp.tknzr.CharTknzr`.

Test target:
- :py:meth:`lmp.tknzr.CharTknzr.__init__`.
"""

import pytest

from lmp.tknzr import CharTknzr


def test_is_uncased():
    r"""``is_uncased`` must be an instance of `bool`."""

    # Test case: Type mismatched.
    wrong_typed_inputs = [
        0, 1, -1, 0.1, '', (), [], {}, set(), None, ..., NotImplemented,
    ]

    for bad_is_uncased in wrong_typed_inputs:
        with pytest.raises(TypeError) as excinfo:
            CharTknzr(
                is_uncased=bad_is_uncased,
                max_vocab=-1,
                min_count=1,
                tk2id=None,
            )

        assert (
            '`is_uncased` must be an instance of `bool`' in str(excinfo.value)
        )

    # Test case: Correct input.
    for good_is_uncased in [False, True]:
        tknzr = CharTknzr(
            is_uncased=good_is_uncased,
            max_vocab=-1,
            min_count=1,
            tk2id=None,
        )
        assert tknzr.is_uncased == good_is_uncased


def test_max_vocab():
    r"""``max_vocab`` must be an integer larger than or equal to ``-1``."""

    # Test case: Type mismatched.
    wrong_typed_inputs = [
        -1.0, 0.0, 1.0, '', (), [], {}, set(), None, ..., NotImplemented,
    ]

    for bad_max_vocab in wrong_typed_inputs:
        with pytest.raises(TypeError) as excinfo:
            CharTknzr(
                is_uncased=True,
                max_vocab=bad_max_vocab,
                min_count=1,
                tk2id=None,
            )

        assert '`max_vocab` must be an instance of `int`' in str(excinfo.value)

    # Test case: Invalid value.
    with pytest.raises(ValueError) as excinfo:
        CharTknzr(
            is_uncased=True,
            max_vocab=-2,
            min_count=1,
            tk2id=None,
        )

    assert (
        '`max_vocab` must be larger than or equal to `-1`'
        in str(excinfo.value)
    )

    # Test case: Correct input.
    for good_max_vocab in range(-1, 10, 1):
        tknzr = CharTknzr(
            is_uncased=True,
            max_vocab=good_max_vocab,
            min_count=1,
            tk2id=None,
        )
        assert tknzr.max_vocab == good_max_vocab


def test_min_count():
    r"""``min_count`` must be an integer larger than ``0``."""

    # Test case: Type mismatched.
    wrong_typed_inputs = [
        -1.0, 0.0, 1.0, '', (), [], {}, set(), None, ..., NotImplemented,
    ]

    for bad_min_count in wrong_typed_inputs:
        with pytest.raises(TypeError) as excinfo:
            CharTknzr(
                is_uncased=True,
                max_vocab=-1,
                min_count=bad_min_count,
                tk2id=None,
            )

        assert '`min_count` must be an instance of `int`' in str(excinfo.value)

    # Test case: Invalid value.
    wrong_value_inputs = [-1, 0]

    for bad_min_count in wrong_value_inputs:
        with pytest.raises(ValueError) as excinfo:
            CharTknzr(
                is_uncased=True,
                max_vocab=-1,
                min_count=bad_min_count,
                tk2id=None,
            )

        assert '`min_count` must be larger than `0`' in str(excinfo.value)

    # Test case: Correct input.
    for good_min_count in range(1, 10):
        tknzr = CharTknzr(
            is_uncased=True,
            max_vocab=-1,
            min_count=good_min_count,
            tk2id=None,
        )
        assert tknzr.min_count == good_min_count


def test_tk2id():
    r"""``tk2id`` must be an dictionary which maps `str` to `int`."""

    # Test case: Type mismatched.
    wrong_typed_inputs = [
        False, True, -1, 0, 1, -1.0, 0.1, '', (), [], set(), ...,
        NotImplemented,
    ]

    for bad_tk2id in wrong_typed_inputs:
        with pytest.raises(TypeError) as excinfo:
            CharTknzr(
                is_uncased=True,
                max_vocab=-1,
                min_count=1,
                tk2id=bad_tk2id,
            )

        assert '`tk2id` must be an instance of `dict`' in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        CharTknzr(
            is_uncased=True,
            max_vocab=-1,
            min_count=1,
            tk2id={1: 1},
        )

    assert (
        'All keys in `tk2id` must be instances of `str`' in str(excinfo.value)
    )

    with pytest.raises(TypeError) as excinfo:
        CharTknzr(
            is_uncased=True,
            max_vocab=-1,
            min_count=1,
            tk2id={'a': 'a'},
        )

    assert (
        'All values in `tk2id` must be instances of `int`'
        in str(excinfo.value)
    )

    # Test case: Invalid value.
    with pytest.raises(ValueError) as excinfo:
        CharTknzr(
            is_uncased=True,
            max_vocab=-1,
            min_count=1,
            tk2id={'a': -1},
        )

    assert (
        'All values in `tk2id` must be non-negative integers'
        in str(excinfo.value)
    )

    # Test case: Correct input.
    good_tk2id = {
        'a': 1,
        'b': 2,
    }
    tknzr = CharTknzr(
        is_uncased=True,
        max_vocab=-1,
        min_count=1,
        tk2id=good_tk2id,
    )
    assert tknzr.tk2id == good_tk2id

    # Test case: Default value.
    tknzr = CharTknzr(
        is_uncased=True,
        max_vocab=-1,
        min_count=1,
        tk2id=None,
    )
    assert tknzr.tk2id == {
        CharTknzr.bos_tk: CharTknzr.bos_tkid,
        CharTknzr.eos_tk: CharTknzr.eos_tkid,
        CharTknzr.pad_tk: CharTknzr.pad_tkid,
        CharTknzr.unk_tk: CharTknzr.unk_tkid,
    }
