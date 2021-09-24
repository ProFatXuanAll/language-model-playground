r"""Setup fixtures for testing :py:mod:`lmp.util`."""

import os

import pytest

import lmp
from lmp.tknzr import BaseTknzr, CharTknzr, WsTknzr
from lmp.model import RNNModel
from lmp.dset import WikiText2Dset


@pytest.fixture
def tknzr() -> BaseTknzr:
    r"""Example tokenizer instance."""

    return CharTknzr(
        is_uncased=True,
        max_vocab=-1,
        min_count=1,
        tk2id={
            CharTknzr.bos_tk: CharTknzr.bos_tkid,
            CharTknzr.eos_tk: CharTknzr.eos_tkid,
            CharTknzr.pad_tk: CharTknzr.pad_tkid,
            CharTknzr.unk_tk: CharTknzr.unk_tkid,
            'a': 4,
            'b': 5,
            'c': 6,
        },
    )


@pytest.fixture
def exp_name():
    return "test_exp_name"


@pytest.fixture
def clean_training_cfg(
    exp_name,
    request,
):
    r"""Clean up the saveing training configuration file."""
    file_dir = os.path.join(lmp.path.EXP_PATH, exp_name)
    file_path = os.path.join(file_dir, lmp.util.cfg.CFG_NAME)

    def fin():
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(file_dir) and not os.listdir(file_dir):
            os.removedirs(file_dir)

    request.addfinalizer(fin)


@pytest.fixture
def clean_dset(
    exp_name,
    request,
):
    r"""Clean up the saving dataset."""
    def remove():
        file_path = os.path.join(
            lmp.path.DATA_PATH,
            WikiText2Dset.file_name.format('valid'),
        )

        if os.path.exists(file_path):
            os.remove(file_path)

        # Remove data directory if it is empty.
        if not os.listdir(lmp.path.DATA_PATH):
            os.removedirs(lmp.path.DATA_PATH)

    request.addfinalizer(remove)


@pytest.fixture
def clean_logger(
    exp_name,
    request,
):
    r"""Clean up the saving tensorboard logger"""
    def remove():
        file_dir = os.path.join(lmp.path.LOG_PATH, exp_name)

        for event_file in os.listdir(file_dir):
            os.remove(
                os.path.join(file_dir, event_file)
            )

        if os.path.exists(file_dir):
            os.removedirs(file_dir)

    request.addfinalizer(remove)


@pytest.fixture
def clean_model(
    exp_name,
    request,
):
    r"""Clean the saving model."""
    abs_dir_path = os.path.join(lmp.path.EXP_PATH, exp_name)

    def remove():
        abs_file_path = os.path.join(
            abs_dir_path, RNNModel.file_name.format(1)
        )
        if os.path.exists(abs_file_path):
            os.remove(abs_file_path)

        abs_file_path = os.path.join(
            abs_dir_path, RNNModel.file_name.format(3)
        )
        if os.path.exists(abs_file_path):
            os.remove(abs_file_path)

        abs_file_path = os.path.join(
            abs_dir_path, RNNModel.file_name.format(5)
        )
        if os.path.exists(abs_file_path):
            os.remove(abs_file_path)

        if os.path.exists(abs_dir_path) and not os.listdir(abs_dir_path):
            os.removedirs(abs_dir_path)

    request.addfinalizer(remove)


@pytest.fixture
def clean_tknzr(
    exp_name,
    request,
):
    r"""Clean up the saving tokenizer."""
    def remove():
        file_dir = os.path.join(lmp.path.EXP_PATH, exp_name)
        file_path = os.path.join(file_dir, WsTknzr.file_name)

        if os.path.exists(file_path):
            os.remove(file_path)

        if os.path.exists(file_dir) and not os.listdir(file_dir):
            os.removedirs(file_dir)

    request.addfinalizer(remove)
