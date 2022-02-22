r"""Use pre-trained language model checkpoint to generate continual text of given text segment.

One must first run the script :py:mod:`lmp.script.train_model` before running this script.  This script use pre-trained
language model checkpoint to generate continual text of given text segment.  Most inference (generation) methods are
stochastic process, only some are deterministic.

See Also
--------
:doc:`lmp.infer </infer/index>`
  All available inference methods.
:doc:`lmp.model </model/index>`
  All available language models.
:doc:`lmp.script.train_model </script/train_model>`
  Train language model.

Examples
--------
The following example use ``"Hello world"`` as conditioned text segment to generate continual text with pre-trained
language model experiment ``my_model_exp``.  It use ``top-1`` inference method to generate continual text.

.. code-block::

   python -m lmp.script.gen_txt top-1 \
     --ckpt 5000 \
     --exp_name my_model_exp \
     --max_seq_len 128 \
     --txt "Hello world"

The following example use the same conditioned text segment but inferencing with ``top-k`` inference method.

.. code-block::

   python -m lmp.script.gen_txt top-1 \
     --ckpt 5000 \
     --exp_name my_model_exp \
     --k 10 \
     --max_seq_len 128 \
     --txt "Hello world"

You can use ``-h`` or ``--help`` options to get a list of available inference methods.

.. code-block:: shell

   python -m lmp.script.gen_txt -h

You can use ``-h`` or ``--help`` options on a specific inference method to get a list of supported CLI arguments.

.. code-block:: shell

   python -m lmp.script.gen_txt top-k -h
"""

import argparse
import gc
import sys
from typing import List

import torch

import lmp.model
import lmp.util.cfg
import lmp.util.infer
import lmp.util.model
import lmp.util.rand
import lmp.util.tknzr
from lmp.infer import INFER_OPTS


def parse_args(argv: List[str]) -> argparse.Namespace:
  """Parse CLI arguments.

  Parameters
  ----------
  argv: list[str]
    List of CLI arguments.

  See Also
  --------
  sys.argv
    Python CLI arguments interface.

  Returns
  -------
  argparse.Namespace
    Parsed CLI arguments.
  """
  # Create parser.
  parser = argparse.ArgumentParser(
    'python -m lmp.script.gen_txt',
    description='Use pre-trained language model checkpoint to generate continual text of given text segment.',
  )

  # Use inference method name to create subparser for all inference methods.
  subparsers = parser.add_subparsers(dest='infer_name', required=True)
  for infer_name, infer_type in INFER_OPTS.items():
    infer_subparser = subparsers.add_parser(infer_name, description=f'Use {infer_type.__name__} as inference method.')

    # Required arguments.
    group = infer_subparser.add_argument_group('language model inference arguments')
    group.add_argument(
      '--ckpt',
      help='Pre-trained language model checkpoint.',
      required=True,
      type=int,
    )
    group.add_argument(
      '--exp_name',
      help='Pre-trained language model experiment name.',
      required=True,
      type=str,
    )
    group.add_argument(
      '--max_seq_len',
      help='Maximum sequence length constraint.',
      required=True,
      type=int,
    )
    group.add_argument(
      '--txt',
      help='Text segment which the generation process is condition on.',
      required=True,
      type=str,
    )

    # Optional arguments.
    group.add_argument(
      '--seed',
      default=42,
      help='Random seed.',
      type=int,
    )

    # Add inference method specific arguments.
    infer_type.add_CLI_args(parser=infer_subparser)

  return parser.parse_args(argv)


def main(argv: List[str]) -> None:
  """Script entry point.

  Parameters
  ----------
  argv: list[str]
    List of CLI arguments.

  Returns
  -------
  None
  """
  # Parse CLI arguments.
  args = parse_args(argv=argv)

  # Set random seed for reproducibility.
  lmp.util.rand.set_seed(seed=args.seed)

  # Load pre-trained model configuration.
  model_cfg = lmp.util.cfg.load(exp_name=args.exp_name)

  # Load pre-trained tokenizer instance.
  tknzr = lmp.util.tknzr.load(exp_name=model_cfg.tknzr_exp_name)

  # Load pre-trained model instance.
  model = lmp.util.model.load(ckpt=args.ckpt, exp_name=args.exp_name)

  # Set model to evaluation model.  This turn off dropout layers in model.
  model = model.eval()

  # Get model running device.
  device = torch.device('cpu')
  if torch.cuda.is_available():
    device = torch.device('cuda')

  # Move model to running device.
  model = model.to(device)

  # Get inference method.
  infer = lmp.util.infer.create(**args.__dict__)

  # Generate text with specified inference method.
  txt = infer.gen(model=model, tknzr=tknzr, txt=args.txt)

  # Output generate text.
  print(txt)

  # Free memory.  This is only need for unit test.
  del args
  del device
  del infer
  del model
  del model_cfg
  del tknzr
  del txt
  torch.cuda.empty_cache()
  gc.collect()


if __name__ == '__main__':
  main(argv=sys.argv[1:])
