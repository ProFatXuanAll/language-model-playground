"""Setup fixtures for testing :py:mod:`lmp`."""

import argparse
import os
import uuid
from typing import Dict, List

import pytest

import lmp.util.path
from lmp.dset import ChPoemDset, WikiText2Dset


def pytest_addoption(parser: argparse.ArgumentParser) -> None:
  """Pytest CLI parser."""
  parser.addoption(
    '--no_cache_dset',
    action='store_false',
    default=False,
    help='Set to true to delete all downloaded datasets after testing.',
  )


@pytest.fixture
def exp_name() -> str:
  """Test experiment name.

  Experiment name is used to save experiment result, such as tokenizer
  configuration, model checkpoint and logging.

  Returns
  -------
  str
      Experiment name with the format ``test-uuid``.
  """
  return 'test-' + str(uuid.uuid4())


@pytest.fixture(
  params=[
    # Full-width to half-width.
    {
      'input': '０',
      'output': '0'
    },
    # NFKD to NFKC.
    {
      'input': 'é',
      'output': 'é'
    },
  ]
)
def nfkc_txt(request) -> Dict[str, str]:
  """Normalize text with NFKC."""
  return request.param


@pytest.fixture(params=[
  {
    'input': 'a  b  c',
    'output': 'a b c'
  },
  {
    'input': '  ',
    'output': ''
  },
])
def ws_collapse_txt(request) -> Dict[str, str]:
  """Collapse consecutive whitespaces."""
  return request.param


@pytest.fixture(
  params=[
    {
      'input': ' abc',
      'output': 'abc'
    },
    {
      'input': 'abc ',
      'output': 'abc'
    },
    {
      'input': ' abc ',
      'output': 'abc'
    },
  ]
)
def ws_strip_txt(request) -> Dict[str, str]:
  """Strip whitespaces at head and tail."""
  return request.param


@pytest.fixture(scope='session')
def ch_poem_file_paths(request) -> List[str]:
  """Chinese poem download file path.

  After testing, clean up files and directories created during test.
  """
  abs_file_paths = [os.path.join(lmp.util.path.DATA_PATH, f'{ver}.csv') for ver in ChPoemDset.vers]

  def fin() -> None:
    # Only delete dataset when explicitly specified by CLI arguments.  This speed up test process.
    if not request.config.getoption('--no_cache_dset'):
      return
    for abs_file_path in abs_file_paths:
      if os.path.exists(abs_file_path):
        os.remove(abs_file_path)
    if os.path.exists(lmp.util.path.DATA_PATH) and not os.listdir(lmp.util.path.DATA_PATH):
      os.removedirs(lmp.util.path.DATA_PATH)

  request.addfinalizer(fin)
  return abs_file_paths


@pytest.fixture(scope='session')
def wiki_text_2_file_paths(request) -> List[str]:
  """Chinese poem download file path.

  After testing, clean up files and directories created during test.
  """
  abs_file_paths = [os.path.join(lmp.util.path.DATA_PATH, f'wiki.{ver}.tokens') for ver in WikiText2Dset.vers]

  def fin() -> None:
    # Only delete dataset when explicitly specified by CLI arguments.  This speed up test process.
    if not request.config.getoption('--no_cache_dset'):
      return

    for abs_file_path in abs_file_paths:
      if os.path.exists(abs_file_path):
        os.remove(abs_file_path)
    if os.path.exists(lmp.util.path.DATA_PATH) and not os.listdir(lmp.util.path.DATA_PATH):
      os.removedirs(lmp.util.path.DATA_PATH)

  request.addfinalizer(fin)
  return abs_file_paths
