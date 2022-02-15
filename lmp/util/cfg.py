"""Save and load training configurations."""

import argparse
import json
import os

import lmp.util.path
import lmp.util.validate

CFG_NAME = 'cfg.json'


def save(args: argparse.Namespace, exp_name: str) -> None:
  """Save training configurations into JSON file.

  Save training configuration in to path ``project_root/exp/exp_name/cfg.json``.  All successfully parsed CLI arguments
  will be saved.  If folders along the saving path do not exist, then this method will create folders recursively.

  .. danger::

    This method overwrite existed files.  Make sure you know what you are doing before calling this method.

  Parameters
  ----------
  args: argparse.Namespace
    Parsed CLI arguments which will be saved.
  exp_name: str
    Name of the training experiment.

  Returns
  -------
  None

  See Also
  --------
  lmp.util.cfg.load
    Load training configurations from JSON file.

  Examples
  --------
  >>> import argparse
  >>> import lmp.util.cfg
  >>> args = argparse.Namespace(a=1, b=2, c=3)
  >>> lmp.util.cfg.save(args=args, exp_name='my_exp')
  None
  """
  # `args` validation.
  lmp.util.validate.raise_if_not_instance(val=args, val_name='args', val_type=argparse.Namespace)

  # `exp_name` validation.
  lmp.util.validate.raise_if_not_instance(val=exp_name, val_name='exp_name', val_type=str)
  lmp.util.validate.raise_if_empty_str(val=exp_name, val_name='exp_name')

  # `file_dir` validation.
  file_dir = os.path.join(lmp.util.path.EXP_PATH, exp_name)
  lmp.util.validate.raise_if_is_file(path=file_dir)

  # `file_path` validation.
  file_path = os.path.join(file_dir, CFG_NAME)
  lmp.util.validate.raise_if_is_directory(path=file_path)

  # Create experiment path if not exist.
  if not os.path.exists(file_dir):
    os.makedirs(file_dir)

  # Save configuration in JSON format.
  with open(file_path, 'w', encoding='utf-8') as output_file:
    json.dump(args.__dict__, output_file, ensure_ascii=False, sort_keys=True)


def load(exp_name: str) -> argparse.Namespace:
  """Load training configuration from JSON file.

  Load training configuration from path ``project_root/exp/exp_name/cfg.json``.  Loaded configurations will be wrapped
  in :py:class:`argparse.Namespace` for convenience.

  Parameters
  ----------
  exp_name: str
    Name of the training experiment.

  Returns
  -------
  argparse.Namespace
    Training experiment's configurations.

  See Also
  --------
  lmp.util.cfg.save
    Save training configurations into JSON file.

  Examples
  --------
  >>> import argparse
  >>> import lmp.util.cfg
  >>> args = argparse.Namespace(a=1, b=2, c=3)
  >>> lmp.util.cfg.save(args=args, exp_name='my_exp')
  None
  >>> args == lmp.util.cfg.load(exp_name='my_exp')
  True
  """
  # `exp_name` validation.
  lmp.util.validate.raise_if_not_instance(val=exp_name, val_name='exp_name', val_type=str)
  lmp.util.validate.raise_if_empty_str(val=exp_name, val_name='exp_name')

  # `file_path` validation.
  file_path = os.path.join(lmp.util.path.EXP_PATH, exp_name, CFG_NAME)
  lmp.util.validate.raise_if_is_directory(path=file_path)

  # Load configuration from JSON file.
  with open(file_path, 'r', encoding='utf-8') as input_file:
    cfg = json.load(input_file)

  # Wrap configuration with `argparse.Namespace` for convenience.
  return argparse.Namespace(**cfg)
