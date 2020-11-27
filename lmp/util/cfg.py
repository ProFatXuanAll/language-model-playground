r"""Save training configuration and load pre-trained configuration."""

import argparse
import json
import os

import lmp.path

CFG_NAME = 'cfg.json'


def save(args: argparse.Namespace, exp_name: str) -> None:
    r"""Save training configuration in JSON format.

    Save training configuration in to path ``exp/exp_name/cfg.json``.
    All CLI arguments will be saved.
    If experiment path ``exp/exp_name/cfg.json`` does not exists, then create
    path recursively.

    Parameters
    ==========
    args: argparse.Namespace
        CLI arguments which will be saved as training configuration.
    exp_name: str
        Current training experiment name.

    Raises
    ======
    FileExistsError
        If experiment path ``exp/exp_name/cfg.json`` exists and is a directory.

    See Also
    ========
    lmp.util.cfg.load

    Examples
    ========
    >>> import argparse
    >>> import lmp.util.cfg
    >>> args = argparse.Namespace(a=1, b=2, c=3)
    >>> lmp.util.cfg.save(args=args, exp_name='my_exp')
    None
    """
    # Get file directory and path.
    file_dir = os.path.join(lmp.path.EXP_PATH, exp_name)
    file_path = os.path.join(file_dir, CFG_NAME)

    # Create experiment path if not exist.
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    if os.path.isdir(file_path):
        raise FileExistsError(f'{file_path} is a directory.')

    # Save configuration in JSON format.
    with open(file_path, 'w', encoding='utf-8') as output_file:
        json.dump(args.__dict__, output_file, ensure_ascii=False)


def load(exp_name: str) -> argparse.Namespace:
    r"""Load pre-trained configuration from JSON file.

    Load pre-trained configuration from path ``exp/exp_name/cfg.json``.
    Experiments must been performed before using this function.
    Wrap configuration in :py:class:`argparse.Namespace` for convenience.

    Parameters
    ==========
    exp_name: str
        Pre-trained experiment name.

    Returns
    =======
    argparse.Namespace
        Pre-trained experiment configuration.

    Raises
    ======
    FileExistsError
        If experiment path ``exp/exp_name/cfg.json`` exists and is a directory.
    FileNotFoundError
        If experiment path ``exp/exp_name/cfg.json`` does not exist.

    See Also
    ========
    lmp.util.cfg.save

    Examples
    ========
    >>> import argparse
    >>> import lmp.util.cfg
    >>> args = argparse.Namespace(a=1, b=2, c=3)
    >>> lmp.util.cfg.save(args=args, exp_name='my_exp')
    None
    >>> args == lmp.util.cfg.load(exp_name='my_exp')
    True
    """
    # Get file path.
    file_path = os.path.join(lmp.path.EXP_PATH, exp_name, CFG_NAME)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f'{file_path} does not exist.')

    if os.path.isdir(file_path):
        raise FileExistsError(f'{file_path} is a directory.')

    # Load configuration from JSON file.
    with open(file_path, 'r', encoding='utf-8') as input_file:
        cfg = json.load(input_file)

    # Wrap configuration with `argparse.Namespace` for convenience.
    return argparse.Namespace(**cfg)
