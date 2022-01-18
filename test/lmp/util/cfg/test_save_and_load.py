"""Test training configuration save and load.

Test target:
- :py:meth:`lmp.util.cfg.load`.
- :py:meth:`lmp.util.cfg.save`.
"""

import argparse
import os

import lmp
import lmp.util.cfg


def test_save_and_load(exp_name: str, cfg_file_path: str) -> None:
  """Must correctly save and load configurations."""
  args = argparse.Namespace(a=1, b=2, c=3)
  lmp.util.cfg.save(args, exp_name)
  assert os.path.exists(cfg_file_path)
  load_args = lmp.util.cfg.load(exp_name)
  assert args == load_args
