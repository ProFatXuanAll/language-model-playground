"""Test the ability to download files.

Test target:
- :py:meth:`lmp.dset.BaseDset.download`.
"""

import os

import pytest

import lmp.util.path
from lmp.dset import BaseDset


@pytest.fixture
def file_path(request) -> str:
  """Download file path.

  After testing, clean up files and directories created during test.
  """
  abs_file_path = os.path.join(lmp.util.path.DATA_PATH, 'README.rst')

  def fin() -> None:
    if os.path.exists(abs_file_path):
      os.remove(abs_file_path)
    if os.path.exists(lmp.util.path.DATA_PATH) and not os.listdir(lmp.util.path.DATA_PATH):
      os.removedirs(lmp.util.path.DATA_PATH)

  request.addfinalizer(fin)
  return abs_file_path


@pytest.fixture
def file_url() -> str:
  """Download target file URL."""
  return 'https://raw.githubusercontent.com/ProFatXuanAll/language-model-playground/main/README.rst'


def test_download_as_text_file(file_path: str, file_url: str) -> None:
  """Must be able to download file and output as text file."""
  BaseDset.download_file(mode='text', download_path=file_path, url=file_url)
  assert os.path.exists(file_path)


def test_download_as_binary_file(file_path: str, file_url: str) -> None:
  """Must be able to download file and output as binary file."""
  BaseDset.download_file(mode='binary', download_path=file_path, url=file_url)
  assert os.path.exists(file_path)
