r"""Train the text-generation model.
Usage:
    python example_train.py --experiment_no 1 --is_uncased True

--experiment_no and --is_uncased are required arguments
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

# self-made modules
import lmp

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()

# Required arguments.
parser.add_argument("--experiment_no", type=int, default=1, required=True, help="using which experiment_no data")

# Optional arguments.
parser.add_argument("--batch_size",         type=int,   default=32,     help="Training batch size.")
parser.add_argument("--dropout",            type=float, default=0,      help="Dropout rate.")
parser.add_argument("--embedding_dim",      type=int,   default=100,    help="Embedding dimension.")
parser.add_argument("--epoch",              type=int,   default=10,     help="Number of training epochs.")
parser.add_argument("--max_norm",           type=float, default=1,      help="Max norm of gradient.Used when cliping gradient norm.")
parser.add_argument("--hidden_dim",         type=int,   default=300,    help="Hidden dimension.")
parser.add_argument("--learning_rate",      type=float, default=1e-4,   help="Optimizer's parameter `lr`.")
parser.add_argument("--min_count",          type=int,   default=0,      help="Minimum of token'sfrequence.")
parser.add_argument("--num_rnn_layers",     type=int,   default=1,      help="Number of rnn layers.")
parser.add_argument("--num_linear_layers",  type=int,   default=2,      help="Number of Linear layers.")
parser.add_argument("--seed",               type=int,   default=7,      help="Control random seed.")
parser.add_argument("--checkpoint",         type=int,   default=500,    help="save model state each check point")
parser.add_argument("--is_uncased",         action="store_true",        help="convert all upper case into lower case.")

args = parser.parse_args()


##############################################
# Hyperparameters setup
##############################################
experiment_no = args.experiment_no
config = lmp.config.BaseConfig(batch_size=args.batch_size,
                               dropout=args.dropout,
                               embedding_dim=args.embedding_dim,
                               epoch=args.epoch,
                               max_norm=args.max_norm,
                               hidden_dim=args.hidden_dim,
                               learning_rate=args.learning_rate,
                               min_count=args.min_count,
                               num_rnn_layers=args.num_rnn_layers,
                               num_linear_layers=args.num_linear_layers,
                               seed=args.seed,
                               is_uncased=args.is_uncased,
                               checkpoint = args.checkpoint)


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

##############################################
# Load data.
##############################################
data_path = os.path.abspath('./data')

df = pd.read_csv(f'{data_path}/news_collection.csv')

##############################################
# Construct tokenizer and perform tokenization.
##############################################
tokenizer = lmp.tokenizer.CharTokenizerByList()



dataset = lmp.dataset.BaseDataset(config=config,
                                  text_list=df['title'],
                                  tokenizer=tokenizer)

data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=config.batch_size,
                                          shuffle=True,
                                          collate_fn=dataset.collate_fn)


##############################################
# Construct RNN model, choose loss function and optimizer.
##############################################
model = lmp.model.LSTMModel(config=config,
                           tokenizer=tokenizer)

model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

##############################################
# train
##############################################
save_path = f'{data_path}/{experiment_no}'
if not os.path.exists(save_path):
    os.mkdir(save_path)

config_save_path = f'{save_path}/config.pickle'
tokenizer_save_path = f'{save_path}/tokenizer.pickle'
model_save_path = f'{save_path}/model.ckpt'

config.save_to_file(config_save_path)

tokenizer.save_to_file(tokenizer_save_path)

model.train()

best_loss = None

model_num = 0
for epoch in range(config.epoch):
    
    print(f'epoch {epoch}')
    total_loss = 0
    iteration_times = 0
    for x, y in tqdm(data_loader, desc='training'):
        x = x.to(device)
        y = y.view(-1).to(device) # shape: (batch_size * sequence_length)
        
        optimizer.zero_grad()
        
        pred_y = model(x)
        pred_y = pred_y.view(-1, tokenizer.vocab_size()) # shape: (batch_size * sequence_length, vocabulary_size)
        loss = criterion(pred_y, y)
        total_loss += float(loss) / len(dataset)

        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_norm)

        optimizer.step()
        iteration_times += 1
        if iteration_times % config.checkpoint == 0:
            torch.save(model.state_dict(), f'{save_path}/model{model_num}.ckpt')
            model_num += 1

    print(f'loss: {total_loss:.10f}')

    if (best_loss is None) or (total_loss < best_loss):
        torch.save(model.state_dict(), model_save_path)
        best_loss = total_loss

print(f'experiment {experiment_no}')
print(f'best loss: {best_loss:.10f}')