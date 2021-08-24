r"""Test save and load operation for model configuration.

Test target:
- :py:meth:`lmp.model._lstm.LSTMModel.load`.
- :py:meth:`lmp.model._lstm.LSTMModel.save`.
"""

import torch


from lmp.model._lstm import LSTMModel


def test_load(model, tknzr, ckpt, exp_name, cleandir):
    r"""Test LSTMModel save and load"""
    model.save(
        ckpt=ckpt,
        exp_name=exp_name,
    )

    gen_model = LSTMModel.load(
        ckpt=ckpt,
        exp_name=exp_name,
        d_emb=1,
        d_hid=1,
        n_hid_lyr=1,
        n_pre_hid_lyr=1,
        n_post_hid_lyr=1,
        p_emb=0.5,
        p_hid=0.5,
        tknzr=tknzr,
    )

    for i, j in zip(model.parameters(), gen_model.parameters()):
        assert torch.equal(i, j)
