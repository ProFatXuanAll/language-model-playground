r"""Train language model.

Tool for training language model on particular dataset.
This script is usually run after training tokenizer.

See Also
========
lmp.model
    All available models.

Examples
========
The following example train :py:class:`lmp.tknzr.WsTknzr` on
:py:class:`lmp.dset.WikiText2Dset` using ``train`` version.

.. code-block:: sh

    python -m lmp.script.train_tokenizer whitespace \
        --dset_name wikitext-2 \
        --exp_name my_exp \
        --max_vocab 10 \
        --min_count 2 \
        --ver train

The training result will be save at ``exp/my_exp``, and can be reused by other
scripts.

One can include more tokens in vocabulary using ``--max_vocab``:

.. code-block:: sh

    python -m lmp.script.train_tokenizer whitespace \
        --dset_name wikitext-2 \
        --exp_name my_exp \
        --max_vocab 10000 \
        --min_count 2 \
        --ver train

Set ``--max_vocab`` to ``-1`` to include all tokens in the dataset:

.. code-block:: sh

    python -m lmp.script.train_tokenizer whitespace \
        --dset_name wikitext-2 \
        --exp_name my_exp \
        --max_vocab -1 \
        --min_count 2 \
        --ver train

Use ``--min_count`` to filter out tokens such as typos, names, locations, etc.

.. code-block:: sh

    python -m lmp.script.train_tokenizer whitespace
        --dset_name wikitext-2 \
        --exp_name my_exp \
        --max_vocab 10000 \
        --min_count 5 \
        --ver train

Use ``--is_uncased`` to avoid differ tokens with same charaters but in
different case.

.. code-block:: sh

    python -m lmp.script.train_tokenizer whitespace
        --dset_name wikitext-2 \
        --exp_name my_exp \
        --is_uncased \
        --max_vocab 10000 \
        --min_count 5 \
        --ver train

Use ``-h`` or ``--help`` options to get list of available tokenizer.

.. code-block:: sh

    python -m lmp.script.train_tokenizer -h
"""

import argparse
import math
import os
from typing import Union

import torch
import torch.optim
import torch.utils.data
import torch.utils.tensorboard
from tqdm import tqdm

import lmp.dset
import lmp.model
import lmp.tknzr
import lmp.util.cfg
import lmp.util.dset
import lmp.util.model
import lmp.util.tknzr

# def train_model(
#         ckpt: int,
#         ckpt_step: int,
#         data_loader: torch.utils.data.DataLoader,
#         epoch: int,
#         experiment: str,
#         max_norm: float,
#         model: Union[lmp.model.BaseRNNModel, lmp.model.BaseResRNNModel],
#         optimizer: Union[torch.optim.SGD, torch.optim.Adam],
#         vocab_size: int
# ) -> None:
#     r"""Helper function for training language model.

#     Continue training from pre-trained ckpt when `ckpt != -1`.

#     Args:
#         ckpt:
#             Pre-trained model's ckpt. Must be bigger than or equal to
#             `-1`.
#         ckpt_step:
#             ckpt save interval. Must be bigger than or equal to `1`.
#         data_loader:
#             `torch.utils.data.DataLoader` for sampling.
#         device:
#             Model running device.
#         epoch:
#             Number of training epoch. Must be bigger than or equal to `1`.
#         experiment:
#             Name of the current experiment. Must not be empty.
#         max_norm:
#             Maximum gradient norm. Must be bigger than `0.0`.
#         model:
#             Language model.
#         optimizer:
#             Language model's optimizer.
#         vocab_size:
#             Number of classes to predict. Must be bigger than or equal to `1`.

#     Raises:
#         TypeError:
#             When one of the arguments are not an instance of their type
#             annotation respectively.
#         ValueError:
#             When one of the arguments do not follow their constraints. See
#             docstring for arguments constraints.
#     """
#     # Type check.
#     if not isinstance(ckpt, int):
#         raise TypeError('`ckpt` must be an instance of `int`.')

#     if not isinstance(ckpt_step, int):
#         raise TypeError('`ckpt_step` must be an instance of `int`.')

#     if not isinstance(data_loader, torch.utils.data.DataLoader):
#         raise TypeError(
#             '`data_loader` must be an instance of '
#             '`torch.utils.data.DataLoader`.'
#         )

#     if not isinstance(device, torch.device):
#         raise TypeError('`device` must be an instance of `torch.device`.')

#     if not isinstance(epoch, int):
#         raise TypeError('`epoch` must be an instance of `int`.')

#     if not isinstance(experiment, str):
#         raise TypeError('`experiment` must be an instance of `str`.')

#     if not isinstance(max_norm, float):
#         raise TypeError('`max_norm` must be an instance of `float`.')

#     if not isinstance(model, (
#             lmp.model.BaseRNNModel,
#             lmp.model.BaseResRNNModel,
#             lmp.model.TransformerModel,
#     )):
#         raise TypeError(
#             '`model` must be an instance of '
#             '`Union[lmp.model.BaseRNNModel, lmp.model.BaseResRNNModel]`.'
#         )

#     if not isinstance(optimizer, (torch.optim.SGD, torch.optim.Adam)):
#         raise TypeError(
#             '`optimizer` must be an instance of '
#             '`Union[torch.optim.SGD, torch.optim.Adam]`.'
#         )

#     if not isinstance(vocab_size, int):
#         raise TypeError('`vocab_size` must be an instance of `int`.')

#     # Value check.
#     if ckpt < -1:
#         raise ValueError('`ckpt` must be bigger than or equal to `-1`.')

#     if ckpt_step < 1:
#         raise ValueError(
#             '`ckpt_step` must be bigger than or equal to `1`.'
#         )

#     if epoch < 1:
#         raise ValueError('`epoch` must be bigger than or equal to `1`.')

#     if not experiment:
#         raise ValueError('`experiment` must not be empty.')

#     if max_norm < 0.0 or math.isnan(max_norm):
#         raise ValueError('`max_norm` must be bigger than `0.0`.')

#     if vocab_size < 1:
#         raise ValueError('`vocab_size` must be bigger than or equal to `1`.')

#     # Set experiment output folder.
#     file_dir = os.path.join(lmp.path.DATA_PATH, experiment)
#     log_dir = os.path.join(lmp.path.DATA_PATH, 'log', experiment)

#     if not os.path.exists(file_dir):
#         os.makedirs(file_dir)

#     if not os.path.exists(log_dir):
#         os.makedirs(log_dir)

#     # Set experiment log folder.
#     writer = torch.utils.tensorboard.SummaryWriter(log_dir)
#     writer.add_graph(model, next(iter(data_loader))[0].to(device))

#     # Define objective function.
#     criterion = torch.nn.CrossEntropyLoss()

#     # Step = number of updates.
#     # Every update must increment `step`.
#     step = 0

#     # Set model to train mode.
#     model.train()

#     # Clean up gradient in model parameters.
#     model.zero_grad()

#     # Initialize total loss.
#     total_loss = 0.0

#     for cur_epoch in range(epoch):

#         epoch_iterator = tqdm(
#             data_loader,
#             desc=f'epoch: {cur_epoch}, loss: {0:.6f}'
#         )

#         for x, y in epoch_iterator:
#             # Increment step for each update.
#             step += 1

#             # Continue training from previous ckpt step.
#             if step < ckpt:
#                 continue

#             # Put tensors on to specified device (CPU or GPU). Reshape `y` into
#             # shape (B x S) for cross-entropy.
#             # x.size = (B, S)
#             # y.size = (B x S)
#             x = x.to(device)
#             y = y.reshape(-1).to(device)

#             # Forward pass.
#             # pred_y_logits.size = (B, S, V)
#             pred_y_logits = model(x)

#             # Reshape `pred_y_logits` into shape (B x S, V) for cross-entropy.
#             pred_y_logits = pred_y_logits.reshape(-1, vocab_size)

#             # Perform cross-entropy.
#             loss = criterion(pred_y_logits, y)

#             # Calculate total loss.
#             total_loss += loss.item()

#             # Log loss.
#             epoch_iterator.set_description(
#                 f'epoch: {cur_epoch}, loss: {loss.item():.6f}'
#             )

#             # Backward pass.
#             loss.backward()

#             # Perform gradient clipping to avoid gradient explosion.
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

#             # Gradient descent.
#             optimizer.step()

#             # `torch` required manually clean up gradient.
#             optimizer.zero_grad()

#             # Save ckpt for each `ckpt_step`.
#             if step % ckpt_step == 0:
#                 torch.save(
#                     model.state_dict(),
#                     os.path.join(file_dir, f'model-{step}.pt')
#                 )
#                 torch.save(
#                     optimizer.state_dict(),
#                     os.path.join(file_dir, f'optimizer-{step}.pt')
#                 )
#                 # Log average loss.
#                 writer.add_scalar('loss', total_loss / ckpt_step, step)
#                 total_loss = 0.0

#     # Save last ckpt.
#     torch.save(
#         model.state_dict(),
#         os.path.join(file_dir, f'model-{step}.pt')
#     )
#     torch.save(
#         optimizer.state_dict(),
#         os.path.join(file_dir, f'optimizer-{step}.pt')
#     )


def parse_arg() -> argparse.Namespace:
    r"""Parse arguments from CLI.

    Argument must begin with a tokenizer name ``tknzr_name``.
    All arguments are added with tokenizer's static method ``train_parser``.

    Returns
    =======
    argparse.Namespace
        Arguments from CLI.
    """
    # Create parser.
    parser = argparse.ArgumentParser(
        'python -m lmp.script.train_tokenizer',
        description='Train tokenizer.',
    )

    # Create subparser for each tokenizer.
    subparsers = parser.add_subparsers(dest='tknzr_name', required=True)

    for tknzr_name, tknzr_clss in lmp.tknzr.TKNZR_OPTS.items():
        # Use tokenizer name as CLI argument.
        tknzr_parser = subparsers.add_parser(
            tknzr_name,
            description=f'Training {tknzr_name} tokenizer.',
        )

        # Add customized training script.
        tknzr_clss.train_parser(tknzr_parser)

    return parser.parse_args()


def main() -> None:
    r"""Script entry point."""
    # Parse command-line argument.
    args = parse_arg()

    # Save training configuration.
    lmp.util.cfg.save(args=args, exp_name=args.exp_name)

    # Get dataset instance with specified version.
    dset = lmp.util.dset.load(dset_name=args.dset_name, ver=args.ver)

    # Get new tokenizer instance.
    tknzr = lmp.util.tknzr.create(**args.__dict__)

    # Build tokenizer's vocabulary.
    tknzr.build_vocab(dset)

    # Save training result.
    tknzr.save(args.exp_name)


if __name__ == '__main__':
    main()
