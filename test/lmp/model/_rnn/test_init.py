r"""Test the construction of :py:class:`lmp.model.RNNModel`.

Test target:
- :py:meth:`lmp.model.RNNModel.__init__`.
"""

import pytest
import torch.nn as nn

from lmp.model import RNNModel
from lmp.tknzr import BaseTknzr


def test_d_emb(tknzr: BaseTknzr):
    r"""``d_emb`` must be an instance of `int` and be positive."""
    # Test case: Type mismatched.
    wrong_typed_inputs = [
        0.0, 0.1, 1.0, '', (), [], {}, set(), None, ..., NotImplemented,
    ]

    for bad_d_emb in wrong_typed_inputs:
        with pytest.raises(TypeError) as excinfo:
            RNNModel(
                d_emb=bad_d_emb,
                d_hid=1,
                n_hid_lyr=1,
                n_pre_hid_lyr=1,
                n_post_hid_lyr=1,
                p_emb=0.0,
                p_hid=0.0,
                tknzr=tknzr,
            )

        assert '`d_emb` must be an instance of `int`' in str(excinfo.value)

    # Test case: Invalid value.
    wrong_value_inputs = [
        0, -1, -2,
    ]

    for bad_d_emb in wrong_value_inputs:
        with pytest.raises(ValueError) as excinfo:
            RNNModel(
                d_emb=bad_d_emb,
                d_hid=1,
                n_hid_lyr=1,
                n_pre_hid_lyr=1,
                n_post_hid_lyr=1,
                p_emb=0.0,
                p_hid=0.0,
                tknzr=tknzr,
            )

        assert (
            '`d_emb` must be bigger than or equal to `1`'
            in str(excinfo.value)
        )


def test_d_hid(tknzr: BaseTknzr):
    r"""``d_hid`` must be an instance of `int` and be positive."""
    # Test case: Type mismatched.
    wrong_typed_inputs = [
        0.0, 0.1, 1.0, '', (), [], {}, set(), None, ..., NotImplemented,
    ]

    for bad_d_hid in wrong_typed_inputs:
        with pytest.raises(TypeError) as excinfo:
            RNNModel(
                d_emb=1,
                d_hid=bad_d_hid,
                n_hid_lyr=1,
                n_pre_hid_lyr=1,
                n_post_hid_lyr=1,
                p_emb=0.0,
                p_hid=0.0,
                tknzr=tknzr,
            )

        assert '`d_hid` must be an instance of `int`' in str(excinfo.value)

    # Test case: Invalid value.
    wrong_value_inputs = [
        0, -1, -2,
    ]

    for bad_d_hid in wrong_value_inputs:
        with pytest.raises(ValueError) as excinfo:
            RNNModel(
                d_emb=1,
                d_hid=bad_d_hid,
                n_hid_lyr=1,
                n_pre_hid_lyr=1,
                n_post_hid_lyr=1,
                p_emb=0.0,
                p_hid=0.0,
                tknzr=tknzr,
            )

        assert (
            '`d_hid` must be bigger than or equal to `1`'
            in str(excinfo.value)
        )


def test_n_hid_lyr(tknzr: BaseTknzr):
    r"""``n_hid_lyr`` must be an instance of `int` and be positive."""
    # Test case: Type mismatched.
    wrong_typed_inputs = [
        0.0, 0.1, 1.0, '', (), [], {}, set(), None, ..., NotImplemented,
    ]

    for bad_n_hid_lyr in wrong_typed_inputs:
        with pytest.raises(TypeError) as excinfo:
            RNNModel(
                d_emb=1,
                d_hid=1,
                n_hid_lyr=bad_n_hid_lyr,
                n_pre_hid_lyr=1,
                n_post_hid_lyr=1,
                p_emb=0.0,
                p_hid=0.0,
                tknzr=tknzr,
            )

        assert '`n_hid_lyr` must be an instance of `int`' in str(excinfo.value)

    # Test case: Invalid value.
    wrong_value_inputs = [
        0, -1, -2,
    ]

    for bad_d_hid_lyr in wrong_value_inputs:
        with pytest.raises(ValueError) as excinfo:
            RNNModel(
                d_emb=1,
                d_hid=1,
                n_hid_lyr=bad_d_hid_lyr,
                n_pre_hid_lyr=1,
                n_post_hid_lyr=1,
                p_emb=0.0,
                p_hid=0.0,
                tknzr=tknzr,
            )

        assert (
            '`n_hid_lyr` must be bigger than or equal to `1`'
            in str(excinfo.value)
        )


def test_n_post_hid_lyr(tknzr: BaseTknzr):
    r"""``n_post_hid_lyr`` must be an instance of `int` and be positive."""
    # Test case: Type mismatched.
    wrong_typed_inputs = [
        0.0, 0.1, 1.0, '', (), [], {}, set(), None, ..., NotImplemented,
    ]

    for bad_n_post_hid_lyr in wrong_typed_inputs:
        with pytest.raises(TypeError) as excinfo:
            RNNModel(
                d_emb=1,
                d_hid=1,
                n_hid_lyr=1,
                n_pre_hid_lyr=1,
                n_post_hid_lyr=bad_n_post_hid_lyr,
                p_emb=0.0,
                p_hid=0.0,
                tknzr=tknzr,
            )

        assert (
            '`n_post_hid_lyr` must be an instance of `int`'
            in str(excinfo.value)
        )

    # Test case: Invalid value.
    wrong_value_inputs = [
        0, -1, -2,
    ]

    for bad_n_post_hid_lyr in wrong_value_inputs:
        with pytest.raises(ValueError) as excinfo:
            RNNModel(
                d_emb=1,
                d_hid=1,
                n_hid_lyr=1,
                n_pre_hid_lyr=1,
                n_post_hid_lyr=bad_n_post_hid_lyr,
                p_emb=0.0,
                p_hid=0.0,
                tknzr=tknzr,
            )

        assert (
            '`n_post_hid_lyr` must be bigger than or equal to `1`'
            in str(excinfo.value)
        )


def test_n_pre_hid_lyr(tknzr: BaseTknzr):
    r"""``n_pre_hid_lyr`` must be an instance of `int` and be positive."""
    # Test case: Type mismatched.
    wrong_typed_inputs = [
        0.0, 0.1, 1.0, '', (), [], {}, set(), None, ..., NotImplemented,
    ]

    for bad_n_pre_hid_lyr in wrong_typed_inputs:
        with pytest.raises(TypeError) as excinfo:
            RNNModel(
                d_emb=1,
                d_hid=1,
                n_hid_lyr=1,
                n_pre_hid_lyr=bad_n_pre_hid_lyr,
                n_post_hid_lyr=1,
                p_emb=0.0,
                p_hid=0.0,
                tknzr=tknzr,
            )

        assert (
            '`n_pre_hid_lyr` must be an instance of `int`'
            in str(excinfo.value)
        )

    # Test case: Invalid value.
    wrong_value_inputs = [
        0, -1, -2,
    ]

    for bad_n_pre_hid_lyr in wrong_value_inputs:
        with pytest.raises(ValueError) as excinfo:
            RNNModel(
                d_emb=1,
                d_hid=1,
                n_hid_lyr=1,
                n_pre_hid_lyr=bad_n_pre_hid_lyr,
                n_post_hid_lyr=1,
                p_emb=0.0,
                p_hid=0.0,
                tknzr=tknzr,
            )

        assert (
            '`n_pre_hid_lyr` must be bigger than or equal to `1`'
            in str(excinfo.value)
        )


def test_p_emb(tknzr: BaseTknzr):
    r"""``p_emb`` must be an instance of `float` and must be a probability."""
    # Test case: Type mismatched.
    wrong_typed_inputs = [
        False, True, 0, 1, '', (), [], {}, set(), None, ..., NotImplemented,
    ]

    for bad_p_emb in wrong_typed_inputs:
        with pytest.raises(TypeError) as excinfo:
            RNNModel(
                d_emb=1,
                d_hid=1,
                n_hid_lyr=1,
                n_pre_hid_lyr=1,
                n_post_hid_lyr=1,
                p_emb=bad_p_emb,
                p_hid=0.0,
                tknzr=tknzr,
            )

        assert '`p_emb` must be an instance of `float`' in str(excinfo.value)

    # Test case: Invalid value.
    wrong_value_inputs = [
        -1.0, -0.1, 1.1, 2.0,
    ]

    for bad_p_emb in wrong_value_inputs:
        with pytest.raises(ValueError) as excinfo:
            RNNModel(
                d_emb=1,
                d_hid=1,
                n_hid_lyr=1,
                n_pre_hid_lyr=1,
                n_post_hid_lyr=1,
                p_emb=bad_p_emb,
                p_hid=0.0,
                tknzr=tknzr,
            )

        assert (
            '`p_emb` must be in the range from `0.0` to `1.0`'
            in str(excinfo.value)
        )


def test_p_hid(tknzr: BaseTknzr):
    r"""``p_hid`` must be an instance of `float` and must be a probability."""
    # Test case: Type mismatched.
    wrong_typed_inputs = [
        False, True, 0, 1, '', (), [], {}, set(), None, ..., NotImplemented,
    ]

    for bad_p_hid in wrong_typed_inputs:
        with pytest.raises(TypeError) as excinfo:
            RNNModel(
                d_emb=1,
                d_hid=1,
                n_hid_lyr=1,
                n_pre_hid_lyr=1,
                n_post_hid_lyr=1,
                p_emb=0.0,
                p_hid=bad_p_hid,
                tknzr=tknzr,
            )

        assert '`p_hid` must be an instance of `float`' in str(excinfo.value)

    # Test case: Invalid value.
    wrong_value_inputs = [
        -1.0, -0.1, 1.1, 2.0,
    ]

    for bad_p_hid in wrong_value_inputs:
        with pytest.raises(ValueError) as excinfo:
            RNNModel(
                d_emb=1,
                d_hid=1,
                n_hid_lyr=1,
                n_pre_hid_lyr=1,
                n_post_hid_lyr=1,
                p_emb=0.0,
                p_hid=bad_p_hid,
                tknzr=tknzr,
            )

        assert (
            '`p_hid` must be in the range from `0.0` to `1.0`'
            in str(excinfo.value)
        )


def test_emb(d_emb: int, rnn_model: RNNModel, tknzr: BaseTknzr):
    r"""Test embedding layer.

    If ``rnn_model`` is an instance of ``RNNModel``, then ``rnn_model.emb``
    must be an instance of `nn.Embedding` with correct shape.
    """
    # Type check.
    assert isinstance(rnn_model.emb, nn.Embedding)

    # Shape validation.
    assert rnn_model.emb.num_embeddings == tknzr.vocab_size
    assert rnn_model.emb.embedding_dim == d_emb
    assert rnn_model.emb.padding_idx == tknzr.pad_tkid


def test_emb_dp(p_emb: float, rnn_model: RNNModel):
    r"""Test embedding dropout layer.

    If ``rnn_model`` is an instance of ``RNNModel``, then ``rnn_model.emb_dp``
    must be an instance of `nn.Dropout` with correct dropout probability.
    """
    # Type check.
    assert isinstance(rnn_model.emb_dp, nn.Dropout)

    # Check dropout probability.
    assert rnn_model.emb_dp.p == p_emb


def test_pre_hid(
    d_emb: int,
    d_hid: int,
    n_pre_hid_lyr: int,
    p_hid: float,
    rnn_model: RNNModel,
):
    r"""Test Feed-Forward layers before hidden layer.

    If ``rnn_model`` is an instance of ``RNNModel``, then ``rnn_model.pre_hid``
    must be an instance of `nn.Sequential`.
    All ``nn.Linear`` layers in ``rnn_model.pre_hid`` must have correct shape.
    All ``nn.Dropout`` layers in ``rnn_model.pre_hid`` must have correct
    dropout probability.
    All activations in ``rnn_model.pre_hid`` must be ``nn.ReLU``.
    ``rnn_model.pre_hid`` must contain at least one ``nn.Linear``, one
    ``nn.ReLU`` and one ``nn.Dropout`` layers.
    """
    # Type check.
    assert isinstance(rnn_model.pre_hid, nn.Sequential)
    assert isinstance(rnn_model.pre_hid[0], nn.Linear)
    assert isinstance(rnn_model.pre_hid[1], nn.ReLU)
    assert isinstance(rnn_model.pre_hid[2], nn.Dropout)

    # Shape validation.
    assert rnn_model.pre_hid[0].in_features == d_emb
    assert rnn_model.pre_hid[0].out_features == d_hid

    # Check dropout probability.
    assert rnn_model.pre_hid[2].p == p_hid

    for i in range(1, n_pre_hid_lyr):
        # Type check.
        assert isinstance(rnn_model.pre_hid[3 * i], nn.Linear)
        assert isinstance(rnn_model.pre_hid[3 * i + 1], nn.ReLU)
        assert isinstance(rnn_model.pre_hid[3 * i + 2], nn.Dropout)

        # Shape validation.
        assert rnn_model.pre_hid[3 * i].in_features == d_hid
        assert rnn_model.pre_hid[3 * i].out_features == d_hid

        # Check dropout probability.
        assert rnn_model.pre_hid[3 * i + 2].p == p_hid


def test_hid(d_hid: int, n_hid_lyr: int, p_hid: float, rnn_model: RNNModel):
    r"""Test hidden layer.

    If ``rnn_model`` is an instance of ``RNNModel``, then ``rnn_model.hid``
    must be an instance of `nn.RNN` with correct shape.
    """
    # Type check.
    assert isinstance(rnn_model.hid, nn.RNN)

    # Shape validation.
    assert rnn_model.hid.batch_first
    if n_hid_lyr == 1:
        assert rnn_model.hid.input_size == d_hid
        assert rnn_model.hid.hidden_size == d_hid
    else:
        assert rnn_model.hid.input_size == d_hid
        assert rnn_model.hid.hidden_size == d_hid
        assert rnn_model.hid.dropout == p_hid


def test_post_hid(rnn_model: RNNModel, d_emb, p_hid, d_hid, n_post_hid_lyr):
    r"""Test Feed-Forward layers after hidden layer.

    If ``rnn_model`` is an instance of ``RNNModel``, then
    ``rnn_model.post_hid`` must be an instance of `nn.Sequential`.
    All ``nn.Linear`` layers in ``rnn_model.post_hid`` must have correct shape.
    All ``nn.Dropout`` layers in ``rnn_model.post_hid`` must have correct
    dropout probability.
    All activations in ``rnn_model.post_hid`` must be ``nn.ReLU``.
    ``rnn_model.post_hid`` must contain at least one ``nn.Dropout`` and one
    ``nn.Linear`` layers.
    """
    # Type check.
    assert isinstance(rnn_model.post_hid, nn.Sequential)
    assert isinstance(rnn_model.post_hid[-2], nn.Dropout)
    assert isinstance(rnn_model.post_hid[-1], nn.Linear)

    # Check dropout probability.
    assert rnn_model.post_hid[-2].p == p_hid

    # Shape validation.
    assert rnn_model.post_hid[-1].in_features == d_hid
    assert rnn_model.post_hid[-1].out_features == d_emb

    for i in range(n_post_hid_lyr - 1):
        # Type check.
        assert isinstance(rnn_model.post_hid[3 * i], nn.Dropout)
        assert isinstance(rnn_model.post_hid[3 * i + 1], nn.Linear)
        assert isinstance(rnn_model.post_hid[3 * i + 2], nn.ReLU)

        # Check dropout probability.
        assert rnn_model.post_hid[3 * i].p == p_hid

        # Shape validation.
        assert rnn_model.post_hid[3 * i + 1].in_features == d_hid
        assert rnn_model.post_hid[3 * i + 1].out_features == d_hid