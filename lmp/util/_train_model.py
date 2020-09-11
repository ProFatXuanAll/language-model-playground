r"""Helper function for training model.

Usage:
    import lmp.util

    lmp.util.train_model(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import os

from typing import Union

# 3rd-party modules

import torch
import torch.nn
import torch.optim
import torch.utils.data
import torch.utils.tensorboard

from tqdm import tqdm

# self-made modules

import lmp.config
import lmp.dataset
import lmp.model
import lmp.path
import lmp.tokenizer


def train_model(
        checkpoint: int,
        checkpoint_step: int,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        epoch: int,
        experiment: str,
        max_norm: float,
        model: Union[
            lmp.model.BaseRNNModel,
            lmp.model.BaseResRNNModel,
            lmp.model.BaseSelfAttentionRNNModel,
            lmp.model.BaseSelfAttentionResRNNModel
        ],
        optimizer: Union[torch.optim.SGD, torch.optim.Adam],
        vocab_size: int
) -> None:
    r"""Helper function for training language model.

    Continue training from pre-trained checkpoint when `checkpoint != -1`.

    Args:
        checkpoint:
            Pre-trained model's checkpoint. Must be bigger than or equal to
            `-1`.
        checkpoint_step:
            Checkpoint save interval. Must be bigger than or equal to `1`.
        data_loader:
            `torch.utils.data.DataLoader` for sampling.
        device:
            Model running device.
        epoch:
            Number of training epoch. Must be bigger than or equal to `1`.
        experiment:
            Name of the current experiment. Must not be empty.
        max_norm:
            Maximum gradient norm. Must be bigger than `0.0`.
        model:
            Language model.
        optimizer:
            Language model's optimizer.
        vocab_size:
            Number of classes to predict. Must be bigger than or equal to `1`.

    Raises:
        TypeError:
            When one of the arguments are not an instance of their type
            annotation respectively.
        ValueError:
            When one of the arguments do not follow their constraints. See
            docstring for arguments constraints.
    """
    # Type check.
    if not isinstance(checkpoint, int):
        raise TypeError('`checkpoint` must be an instance of `int`.')

    if not isinstance(checkpoint_step, int):
        raise TypeError('`checkpoint_step` must be an instance of `int`.')

    if not isinstance(data_loader, torch.utils.data.DataLoader):
        raise TypeError(
            '`data_loader` must be an instance of '
            '`torch.utils.data.DataLoader`.'
        )

    if not isinstance(device, torch.device):
        raise TypeError('`device` must be an instance of `torch.device`.')

    if not isinstance(epoch, int):
        raise TypeError('`epoch` must be an instance of `int`.')

    if not isinstance(experiment, str):
        raise TypeError('`experiment` must be an instance of `str`.')

    if not isinstance(max_norm, float):
        raise TypeError('`max_norm` must be an instance of `float`.')

    if not isinstance(model, (
            lmp.model.BaseRNNModel,
            lmp.model.BaseResRNNModel,
            lmp.model.BaseSelfAttentionRNNModel,
            lmp.model.BaseSelfAttentionResRNNModel
    )):
        raise TypeError(
            '`model` must be an instance of '
            '`Union[lmp.model.BaseRNNModel, '
            'lmp.model.BaseResRNNModel, '
            'lmp.model.BaseSelfAttentionRNNModel, '
            'lmp.model.BaseSelfAttentionResRNNModel]`.'
        )

    if not isinstance(optimizer, (torch.optim.SGD, torch.optim.Adam)):
        raise TypeError(
            '`optimizer` must be an instance of '
            '`Union[torch.optim.SGD, torch.optim.Adam]`.'
        )

    if not isinstance(vocab_size, int):
        raise TypeError('`vocab_size` must be an instance of `int`.')

    # Value check.
    if checkpoint < -1:
        raise ValueError('`checkpoint` must be bigger than or equal to `-1`.')

    if checkpoint_step < 1:
        raise ValueError(
            '`checkpoint_step` must be bigger than or equal to `1`.'
        )

    if epoch < 1:
        raise ValueError('`epoch` must be bigger than or equal to `1`.')

    if not experiment:
        raise ValueError('`experiment` must not be empty.')

    if max_norm < 0.0 or math.isnan(max_norm):
        raise ValueError('`max_norm` must be bigger than `0.0`.')

    if vocab_size < 1:
        raise ValueError('`vocab_size` must be bigger than or equal to `1`.')

    # Set experiment output folder.
    file_dir = os.path.join(lmp.path.DATA_PATH, experiment)
    log_dir = os.path.join(lmp.path.DATA_PATH, 'log', experiment)

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Set experiment log folder.
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)

    # Define objective function.
    criterion = torch.nn.CrossEntropyLoss()

    # Step = number of updates.
    # Every update must increment `step`.
    step = 0

    # Set model to train mode.
    model.train()

    # Clean up gradient in model parameters.
    model.zero_grad()

    # Initialize total loss.
    total_loss = 0.0

    for cur_epoch in range(epoch):

        epoch_iterator = tqdm(
            data_loader,
            desc=f'epoch: {cur_epoch}, loss: {0:.6f}'
        )

        for x, y in epoch_iterator:
            # Increment step for each update.
            step += 1

            # Continue training from previous checkpoint step.
            if step < checkpoint:
                continue

            # Put tensors on to specified device (CPU or GPU). Reshape `y` into
            # shape (B x S) for cross-entropy.
            # x.size = (B, S)
            # y.size = (B x S)
            x = x.to(device)
            y = y.reshape(-1).to(device)

            # Forward pass.
            # pred_y_logits.size = (B, S, V)
            pred_y_logits = model(x)

            # Reshape `pred_y_logits` into shape (B x S, V) for cross-entropy.
            pred_y_logits = pred_y_logits.reshape(-1, vocab_size)

            # Perform cross-entropy.
            loss = criterion(pred_y_logits, y)

            # Calculate total loss.
            total_loss += loss.item()

            # Log loss.
            epoch_iterator.set_description(
                f'epoch: {cur_epoch}, loss: {loss.item():.6f}'
            )

            # Backward pass.
            loss.backward()

            # Perform gradient clipping to avoid gradient explosion.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            # Gradient descent.
            optimizer.step()

            # `torch` required manually clean up gradient.
            optimizer.zero_grad()

            # Save checkpoint for each `checkpoint_step`.
            if step % checkpoint_step == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(file_dir, f'model-{step}.pt')
                )
                torch.save(
                    optimizer.state_dict(),
                    os.path.join(file_dir, f'optimizer-{step}.pt')
                )
                # Log average loss.
                writer.add_scalar('loss', total_loss / checkpoint_step, step)
                total_loss = 0.0

    # Save last checkpoint.
    torch.save(
        model.state_dict(),
        os.path.join(file_dir, f'model-{step}.pt')
    )
    torch.save(
        optimizer.state_dict(),
        os.path.join(file_dir, f'optimizer-{step}.pt')
    )
