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

##############################################
# Hyperparameters setup
##############################################
experiment_no = 1
config = lmp.config.BaseConfig(batch_size=32,
                               dropout=0,
                               embedding_dim=100,
                               epoch=3,
                               max_norm=1,
                               hidden_dim=300,
                               learning_rate=10e-4,
                               min_count=0,
                               num_rnn_layers=1,
                               num_linear_layers=2,
                               seed=7)

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
tokenizer = lmp.tokenizer.CharTokenizer()

# 讓使用者決定是否 uncase
parser = argparse.ArgumentParser()
parser.add_argument("-u", "--uncase", help="regard capital letter and lowercase letter as same word",
                    action="store_true")
args = parser.parse_args()

dataset = lmp.dataset.BaseDataset(config=config,
                                  text_list=df['title'],
                                  tokenizer=tokenizer,
                                  is_uncased=args.uncase)

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

best_loss = None
for epoch in range(config.epoch):
    model.train()
    print(f'epoch {epoch}')
    total_loss = 0
    for x, y in tqdm(data_loader, desc='training'):
        x = x.to(device)
        y = y.view(-1).to(device)

        pred_y = model(x)
        pred_y = pred_y.view(-1, tokenizer.vocab_size())
        loss = criterion(pred_y, y)
        total_loss += float(loss) / len(dataset)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_norm)

        optimizer.step()

    print(f'loss: {total_loss:.10f}')

    if (best_loss is None) or (total_loss < best_loss):
        torch.save(model.state_dict(), model_save_path)
        best_loss = total_loss

print(f'experiment {experiment_no}')
print(f'best loss: {best_loss:.10f}')