r"""Test save and load operation for model configuration.

Test target:
- :py:meth:`lmp.model._gru.GRUModel.load`.
- :py:meth:`lmp.model._gru.GRUModel.save`.
"""

import torch


from lmp.model._gru import GRUModel


def test_load(model, tknzr, ckpt, exp_name, cleandir):
    r"""Test GRUModel save and load"""
    model.save(
        ckpt=ckpt,
        exp_name=exp_name,
    )

    gen_model = GRUModel.load(
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
