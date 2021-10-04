r"""Setup fixtures for testing :py:mod:`lmp.util.log`."""

import os

import pytest

import lmp.path


@pytest.fixture
def clean_logger(exp_name: str, request):
    r"""Clean up tensorboard loggings."""
    def remove():
        file_dir = os.path.join(lmp.path.LOG_PATH, exp_name)

        for event_file in os.listdir(file_dir):
            os.remove(os.path.join(file_dir, event_file))

        if os.path.exists(file_dir):
            os.removedirs(file_dir)

    request.addfinalizer(remove)
