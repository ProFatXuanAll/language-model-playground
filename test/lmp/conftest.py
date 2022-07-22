"""Setup fixtures for testing :py:mod:`lmp`."""

import os
import uuid
from typing import Callable, Dict, List

import pytest
import torch

#######################################################################################################################
# Common fixtures.
#######################################################################################################################


@pytest.fixture
def clean_dir_finalizer_factory() -> Callable[[str], None]:
  """Create finalizer function."""

  def create_finalizer(abs_dir_path: str) -> None:

    def finalizer() -> None:
      if not os.path.exists(abs_dir_path):
        return

      # Remove files in the directory.
      for file_name in os.listdir(abs_dir_path):
        try:
          os.remove(os.path.join(abs_dir_path, file_name))
        except Exception:
          pass

      # Remove empty directory.
      if not os.listdir(abs_dir_path):
        os.removedirs(abs_dir_path)

    return finalizer

  return create_finalizer


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


@pytest.fixture
def seed() -> int:
  """Random seed."""
  return 42


#######################################################################################################################
# Inference method related fixtures.
#######################################################################################################################


@pytest.fixture
def k() -> int:
  """``k`` in the top-K."""
  return 5


@pytest.fixture
def p() -> float:
  """``p`` in the top-."""
  return 0.9


#######################################################################################################################
# Model related fixtures.
#######################################################################################################################


@pytest.fixture
def batch_size() -> int:
  """Batch size."""
  return 128


@pytest.fixture
def beta1() -> float:
  """Mock beta1."""
  return 0.9


@pytest.fixture
def beta2() -> float:
  """Mock beta2."""
  return 0.99


@pytest.fixture
def ckpt() -> int:
  """Checkpoint number."""
  return 0


@pytest.fixture
def ckpts() -> List[int]:
  """Model checkpoints."""
  return [0, 1, 2]


@pytest.fixture
def ckpt_step() -> int:
  """Checkpoint step."""
  return 10


@pytest.fixture
def ctx_win(max_seq_len: int) -> int:
  """Context window size."""
  return max_seq_len // 2


@pytest.fixture(params=[1, 2])
def d_blk(request) -> int:
  """Mock memory cell block dimension."""
  return request.param


@pytest.fixture(params=[1, 2])
def d_emb(request) -> int:
  """Mock embedding dimension."""
  return request.param


@pytest.fixture(params=[1, 2])
def d_hid(request) -> int:
  """Mock hidden dimension."""
  return request.param


@pytest.fixture
def eps() -> float:
  """Mock eps."""
  return 1e-8


@pytest.fixture
def host_name() -> str:
  """Mock host name."""
  return '127.0.0.1'


@pytest.fixture
def host_port() -> int:
  """Mock host port."""
  return 42069


@pytest.fixture
def log_step() -> int:
  """Log step."""
  return 10


@pytest.fixture(params=[1e-3, 1e-5])
def lr(request) -> float:
  """Mock learning rate."""
  return request.param


@pytest.fixture
def max_norm() -> float:
  """Gradient clipping max norm."""
  return 1.0


@pytest.fixture
def max_seq_len() -> int:
  """Mock maximum sequence length."""
  return 8


@pytest.fixture(params=[1, 2])
def n_blk(request) -> int:
  """Mock number of memory cell blocks."""
  return request.param


@pytest.fixture(params=[0.1, 0.5])
def p_emb(request) -> float:
  """Mock embedding dropout probability."""
  return request.param


@pytest.fixture(params=[0.1, 0.5])
def p_hid(request) -> float:
  """Mock hidden units dropout probability."""
  return request.param


@pytest.fixture
def total_step(warmup_step: int) -> float:
  """Mock total step."""
  return warmup_step * 2


@pytest.fixture
def warmup_step(ckpt_step: int) -> float:
  """Mock warm up step."""
  return ckpt_step * 10


@pytest.fixture
def wd() -> float:
  """Mock weight decay."""
  return 1e-2


@pytest.fixture
def world_size() -> int:
  """Mock world size."""
  if torch.cuda.device_count() > 1:
    return 2
  return 1


#######################################################################################################################
# Tokenizer related fixtures.
#######################################################################################################################


@pytest.fixture(params=[False, True])
def is_uncased(request) -> bool:
  """Respect cases if set to ``False``."""
  return request.param


@pytest.fixture(params=[-1, 100])
def max_vocab(request) -> int:
  """Maximum vocabulary size."""
  return request.param


@pytest.fixture(params=[0, 10])
def min_count(request) -> int:
  """Minimum token occurrence counts."""
  return request.param


#######################################################################################################################
# Text fixtures.
#######################################################################################################################


@pytest.fixture(
  params=[
    # Convert full-width character to half-width character.
    {
      'input': '０',
      'output': '0'
    },
    # Normalize NFKD character to NFKC character.
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


@pytest.fixture(params=[
  {
    'input': 'ABC',
    'output': 'abc'
  },
  {
    'input': 'abc',
    'output': 'abc'
  },
])
def uncased_txt(request) -> Dict[str, str]:
  """Case-insensitive text."""
  return request.param
