r"""Train the text-generation model.
Usage:
    python example_train.py --experiment_no 1 --is_uncased True

--experiment_no is a required argument.
Run 'python example_train.py --help' for help
"""

# built-in modules
import os

# 3rd-party modules
import numpy as np
import pandas as pd
import torch
import torch.nn
import torch.utils.data
import torch.nn.utils.rnn
import torch.optim
import sklearn.metrics
from tqdm import tqdm
import argparse
import torch.utils.tensorboard

# self-made modules
import lmp



def train_model(config, tokenizer, dataset, model, optimizer, save_path, checkpoint):
    collate_fn = lmp.dataset.BaseDataset.creat_collate_fn(tokenizer=tokenizer,
                                                          max_seq_len=args.max_seq_len)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=config.batch_size,
                                              shuffle=True,
                                              collate_fn=collate_fn)

    criterion = torch.nn.CrossEntropyLoss()

    start_step, start_epoch = 0, 0

    if checkpoint > 0:
        state_path = f'{save_path}/checkpoint{checkpoint}.pt'
        checkpoint_state = torch.load(state_path)

        start_epoch = checkpoint_state['epoch'] + 1
        start_step = checkpoint_state['step'] + 1
    else:
        config.save_to_file(f'{save_path}/config.pickle')
        tokenizer.save_to_file(f'{save_path}/tokenizer.pickle')

    model.train()

    step = start_step

    writer = torch.utils.tensorboard.SummaryWriter(
        f'{save_path}/text-generation-log')

    for epoch in range(start_epoch, config.epoch):

        print(f'epoch {epoch}')
        total_loss = 0

        mini_batch_iterator = tqdm(data_loader)

        for x, y in mini_batch_iterator:

            x = x.to(device)
            y = y.view(-1).to(device)  # shape: (batch_size * sequence_length)
            optimizer.zero_grad()

            pred_y = model(x)
            # shape: (batch_size * sequence_length, vocabulary_size)
            pred_y = pred_y.view(-1, tokenizer.vocab_size())
            loss = criterion(pred_y, y)
            total_loss += loss.item() / len(dataset)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_norm)

            optimizer.step()
            step += 1
            if step % config.checkpoint_step == 0:
                state_param = {'model': model.state_dict(
                ), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'step': step}
                torch.save(
                    state_param, f'{save_path}/checkpoint{step//config.checkpoint_step}.pt')

            writer.add_scalar('text-generation/dataset/Loss',
                              loss.item(),
                              step
                              )

            mini_batch_iterator.set_description(
                f'epoch: {epoch}, loss: {loss.item():.6f} training:'
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required arguments.
    parser.add_argument(
        "--experiment_no",
        type=int,
        default=1,
        required=True,
        help="Current experiment name."
    )

    # Optional arguments.
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size."
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=-1,
        help="Start from specific checkpoint."
    )
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        default=500,
        help="Checkpoint save interval."
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0,
        help="Dropout rate."
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=100,
        help="Embedding dimension."
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=10,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=300,
        help="Hidden dimension."
    )
    parser.add_argument(
        "--is_uncased",
        action="store_true",
        help="Whether to convert text from upper cases to lower cases."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Gradient decent learning rate."
    )
    parser.add_argument(
        "--max_norm",
        type=float,
        default=1,
        help="Gradient bound to avoid gradient explosion."
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=-1,
        help="Text sample max length."
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=0,
        help="Filter out tokens occur less than `min_count`."
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default='lstm',
        help="Language model classes."
    )
    parser.add_argument(
        "--num_rnn_layers",
        type=int,
        default=1,
        help="Number of rnn layers."
    )
    parser.add_argument(
        "--num_linear_layers",
        type=int,
        default=2,
        help="Number of Linear layers."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Control random seed."
    )
    parser.add_argument(
        "--tokenizer_class",
        type=str,
        default='list',
        help="Tokenizer classes."
    )

    args = parser.parse_args()

    project_root = os.path.abspath(f'{os.path.abspath(__file__)}/..')
    data_path = os.path.abspath(f'{project_root}/data')
    # set saved path
    save_path = f'{data_path}/{args.experiment_no}'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    config_save_path = f'{save_path}/config.pickle'
    tokenizer_save_path = f'{save_path}/tokenizer.pickle'

    ##############################################
    # Hyperparameters setup
    ##############################################
    config = lmp.util.load_config(
        args,
        file_path=config_save_path)

    ##############################################
    # Initialize random seed.
    ##############################################
    device = torch.device('cpu')
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Load data.
    df = pd.read_csv(f'{data_path}/news_collection.csv')

    tokenizer = lmp.util.load_tokenizer_by_config(
        config=config,
        checkpoint=args.checkpoint,
        file_path=tokenizer_save_path,
        sentneces=df['title'])

    dataset = lmp.dataset.BaseDataset(text_list=df['title'])

    model = lmp.util.load_model_for_train(
        checkpoint=args.checkpoint,
        config=config,
        device=device,
        save_path=save_path,
        tokenizer=tokenizer
    )

    optimizer = lmp.util.load_optimizer(
        checkpoint=args.checkpoint,
        config=config,
        model=model,
        save_path=save_path)

    print(project_root)

    train_model(
        config=config,
        tokenizer=tokenizer,
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        save_path=save_path,
        checkpoint=args.checkpoint)
