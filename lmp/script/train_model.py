r"""Train language model.

Tool for training language model on particular dataset.
This script is usually run after training tokenizer.
Training performance will be shown on both CLI and tensorboard.
Use ``pipenv run tensorboard`` to launch tensorboard and use browser to open
URL http://localhost:6006/ to see training performance.

See Also
========
lmp.model
    All available models.

Examples
========
The following example train :py:class:`lmp.model.RNNModel` (``RNN``) on
:py:class:`lmp.dset.WikiText2Dset` using ``train`` version
(``--dset_name wikitext-2`` and ``--ver train``).

.. code-block:: sh

    python -m lmp.script.train_model RNN \
        --batch_size 32 \
        --beta1 0.9 \
        --beta2 0.99 \
        --ckpt_step 1000 \
        --dset_name wikitext-2 \
        --eps 1e-8 \
        --exp_name my_model_exp \
        --log_step 200 \
        --lr 1e-4 \
        --max_norm 1 \
        --max_seq_len -1 \
        --n_epoch 10 \
        --tknzr_exp_name my_tknzr_exp \
        --ver train \
        --d_emb 100 \
        --d_hid 300 \
        --n_hid_lyr 2 \
        --n_post_hid_lyr 2 \
        --n_pre_hid_lyr 2 \
        --p_emb 0.1 \
        --p_hid 0.1 \
        --wd 1e-2

The training result will be save at ``exp/my_model_exp``, and can be reused by
other scripts.
We only save checkpoint for each ``--ckpt_step`` step and log performance for
each ``--log_step``.

One can train more epochs by increasing ``--n_epoch``, but be careful model
might be overfitting if trained to much epochs.

.. code-block:: sh

    python -m lmp.script.train_model RNN \
        --batch_size 32 \
        --beta1 0.9 \
        --beta2 0.99 \
        --ckpt_step 1000 \
        --dset_name wikitext-2 \
        --eps 1e-8 \
        --exp_name my_model_exp \
        --log_step 200 \
        --lr 1e-4 \
        --max_norm 1 \
        --max_seq_len -1 \
        --n_epoch 100 \
        --tknzr_exp_name my_tknzr_exp \
        --ver train \
        --d_emb 100 \
        --d_hid 300 \
        --n_hid_lyr 2 \
        --n_post_hid_lyr 2 \
        --n_pre_hid_lyr 2 \
        --p_emb 0.1 \
        --p_hid 0.1 \
        --wd 1e-2

One can reduce overfitting with the following way:

- Increase ``--batch_size`` which makes samples more dynamic.
- Increase ``--wd`` which makes L2 penalty larger as weight grows.
- Reduce model parameters (In :py:class:`lmp.model.RNNModel` this means
  ``--d_emb``, ``--d_hid``, ``n_hid_lyr``, ``n_post_hid_lyr`` and
  ``n_pre_hid_lyr``).
- Use dropout (In :py:class:`lmp.model.RNNModel` this means ``--p_emb`` and
  ``--p_hid``).
- Use any combinations of above tricks.

.. code-block:: sh

    python -m lmp.script.train_model RNN \
        --batch_size 32 \
        --beta1 0.9 \
        --beta2 0.99 \
        --ckpt_step 1000 \
        --dset_name wikitext-2 \
        --eps 1e-8 \
        --exp_name my_model_exp \
        --log_step 200 \
        --lr 1e-4 \
        --max_norm 1 \
        --max_seq_len -1 \
        --n_epoch 10 \
        --tknzr_exp_name my_tknzr_exp \
        --ver train \
        --d_emb 50 \
        --d_hid 100 \
        --n_hid_lyr 1 \
        --n_post_hid_lyr 1 \
        --n_pre_hid_lyr 1 \
        --p_emb 0.5 \
        --p_hid 0.5 \
        --wd 1e-1

We use :py:class:`torch.optim.AdamW` to perform optimization.
Use ``--beta1``, ``--beta2``, ``--eps``, ``--lr`` and ``--wd`` to adjust
optimizer hyper-parameters.
We also use ``--max_norm`` to avoid gradient explosion.

.. code-block:: sh

    python -m lmp.script.train_model RNN \
        --batch_size 32 \
        --beta1 0.95 \
        --beta2 0.98 \
        --ckpt_step 1000 \
        --dset_name wikitext-2 \
        --eps 1e-6 \
        --exp_name my_model_exp \
        --log_step 200 \
        --lr 5e-4 \
        --max_norm 5 \
        --max_seq_len -1 \
        --n_epoch 10 \
        --tknzr_exp_name my_tknzr_exp \
        --ver train \
        --d_emb 100 \
        --d_hid 300 \
        --n_hid_lyr 2 \
        --n_post_hid_lyr 2 \
        --n_pre_hid_lyr 2 \
        --p_emb 0.1 \
        --p_hid 0.1 \
        --wd 1e-2

Use ``-h`` or ``--help`` options to get list of available models.

.. code-block:: sh

    python -m lmp.script.train_model -h
"""

import argparse

import torch
import torch.nn.utils
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
    model = lmp.util.model.create(tknzr=tknzr, **args.__dict__)
    model = model.train()

    # Move model to running device.
    model = model.to(device)

    # Remove weight decay on bias and layer-norm.
    no_decay = ['bias', 'LayerNorm.weight']
    optim_group_params = [
        {
            'params': [
                param for name, param in model.named_parameters()
                if not any(nd in name for nd in no_decay)
            ],
            'weight_decay': args.wd,
        },
        {
            'params': [
                param for name, param in model.named_parameters()
                if any(nd in name for nd in no_decay)
            ],
            'weight_decay': 0.0,
        },
    ]

    # Get new optimizer instance.
    optim = torch.optim.AdamW(
        optim_group_params,
        betas=(args.beta1, args.beta2),
        lr=args.lr,
        eps=args.eps,
    )

    # Get tensorboard logger instance.
    writer = lmp.util.log.get_tb_logger(exp_name=args.exp_name)

    # Log performance target.
    pre_avg_loss = 0.0
    avg_loss = 0.0

    # Global optimization step.
    step = 0

    for epoch in range(args.n_epoch):
        tqdm_dldr = tqdm(
            dldr,
            desc=f'epoch: {epoch}, loss: {pre_avg_loss:.6f}',
        )
        for batch_txt in tqdm_dldr:
            # Encode batch text into batch token ids.
            batch_tkids = tknzr.batch_enc(
                batch_txt=batch_txt,
                max_seq_len=args.max_seq_len,
            )

            # Convert batch token ids to `torch.Tensor` with
            # `dtype == torch.int64`.
            batch_tkids = torch.LongTensor(batch_tkids)

            # Move tensors to model running device.
            batch_tkids = batch_tkids.to(device)

            # Format batch token ids to satisfy language model training format.
            batch_prev_tkids = batch_tkids[..., :-1]
            batch_next_tkids = batch_tkids[..., 1:]

            # Calculate loss using loss function.
            loss = model.loss_fn(
                batch_next_tkids=batch_next_tkids,
                batch_prev_tkids=batch_prev_tkids,
            )

            # Accumulate average loss.
            avg_loss += loss.item()

            # Backward pass / back propagation.
            loss.backward()

            # Perform gradient clipping to avoid gradient explosion.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.max_norm,
            )

            # Gradient descent.
            optim.step()

            # Clean up gradient.
            # This is needed only in `torch`.
            optim.zero_grad()

            # Increment global step.
            step += 1

            # Save checkpoint for each `ckpt_step` step.
            if step % args.ckpt_step == 0:
                model.save(ckpt=step, exp_name=args.exp_name)

            # Log performance for each `log_step` step.
            if step % args.log_step == 0:
                avg_loss = avg_loss / args.log_step

                # Log on CLI.
                tqdm_dldr.set_description(
                    f'epoch: {epoch}, loss: {avg_loss:.6f}',
                )

                # Log on tensorboard
                writer.add_scalar(
                    f'loss/{args.dset_name}/{args.ver}',
                    avg_loss,
                    step,
                )

                # Refresh log performance.
                pre_avg_loss = avg_loss
                avg_loss = 0.0

    # Save last checkpoint.
    model.save(ckpt=step, exp_name=args.exp_name)

    # Close tensorboard logger.
    writer.close()


if __name__ == '__main__':
    main()
