"""Test the ability to download files.

Test target:
- :py:meth:`lmp.dset._base.BaseDset.download`.
"""

import os
from typing import Callable

import pytest

import lmp.dset._base
import lmp.vars


@pytest.fixture
def file_url() -> str:
  """Download target file URL."""
  return 'https://raw.githubusercontent.com/ProFatXuanAll/language-model-playground/main/README.rst'


@pytest.fixture
def file_path(clean_dir_finalizer_factory: Callable[[str], None], exp_name: str, file_url: str, request) -> str:
  """Download file path.

  After testing, clean up files and directories created during test.
  """
  # Create temporary directory.
  abs_dir_path = os.path.join(lmp.vars.DATA_PATH, exp_name)

  if not os.path.exists(abs_dir_path):
    os.makedirs(abs_dir_path)

  abs_file_path = os.path.join(abs_dir_path, file_url.split(r'/')[-1])
  request.addfinalizer(clean_dir_finalizer_factory(abs_dir_path))
  return abs_file_path


def test_download_as_text_file(file_path: str, file_url: str) -> None:
  """Must be able to download file and output as text file."""
  lmp.dset._base.BaseDset.download_file(mode='text', download_path=file_path, url=file_url)
  assert os.path.exists(file_path)


def test_download_as_binary_file(file_path: str, file_url: str) -> None:
  """Must be able to download file and output as binary file."""
  lmp.dset._base.BaseDset.download_file(mode='binary', download_path=file_path, url=file_url)
  assert os.path.exists(file_path)
