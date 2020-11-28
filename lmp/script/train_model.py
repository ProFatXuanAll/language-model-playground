r"""Train language model.

Tool for training language model on particular dataset.
This script is usually run after training tokenizer.

See Also
========
lmp.model
    All available models.

Examples
========
The following example train :py:class:`lmp.model.Wsmodel` on
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
import lmp.util.cfg
import lmp.util.dset
import lmp.util.log
import lmp.util.model
import lmp.util.rand
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

    Argument must begin with a model name ``model_name``.
    All arguments are added with model's static method ``train_parser``.

    Returns
    =======
    argparse.Namespace
        Arguments from CLI.
    """
    # Create parser.
    parser = argparse.ArgumentParser(
        'python -m lmp.script.train_model',
        description='Train language model.',
    )

    # Create subparser for each model.
    subparsers = parser.add_subparsers(dest='model_name', required=True)

    for model_name, model_clss in lmp.model.MODEL_OPTS.items():
        # Use model name as CLI argument.
        model_parser = subparsers.add_parser(
            model_name,
            description=f'Training {model_name} language model.',
        )

        # Add customized arguments.
        model_clss.train_parser(model_parser)

    return parser.parse_args()


def main() -> None:
    r"""Script entry point."""
    # Parse command-line argument.
    args = parse_arg()

    # Save training configuration.
    lmp.util.cfg.save(args=args, exp_name=args.exp_name)

    # Set random seed for reproducibility.
    lmp.util.rand.set_seed(seed=args.seed)

    # Get dataset instance with specified version.
    dset = lmp.util.dset.load(dset_name=args.dset_name, ver=args.ver)

    # Mini-batch random sampler.
    dldr = torch.utils.data.DataLoader(
        dataset=dset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Load pre-trained tokenizer.
    tknzr_cfg = lmp.util.cfg.load(exp_name=args.tknzr_exp_name)
    tknzr = lmp.util.tknzr.load(
        exp_name=args.tknzr_exp_name,
        tknzr_name=tknzr_cfg.tknzr_name,
    )

    # Get model running device.
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    # Get new model instance.
    model = lmp.util.model.create(**args.__dict__)

    # Get tensorboard logger instance.
    writer = lmp.util.log.get_tb_logger(exp_name=args.exp_name)


if __name__ == '__main__':
    main()
