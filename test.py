import os
import pickle
import pandas as pd
import torch

import char_rnn

#####################################################################
# 檔案路徑設定
#####################################################################
data_path = 'data'

train_file = '{}/old-newspaper.tsv'.format(data_path)
train_preprocess_file = '{}/train.preprocess.pickle'.format(data_path)
train_converter_file = '{}/train.converter.pickle'.format(data_path)
train_model_file = '{}/train.model.ckpt'.format(data_path)

#####################################################################
# 讀檔
# 資料集為各國新聞內容且長短不一
# 所以只取語言為「繁體中文」且長度介於 60 ~ 200 之間的文章
#####################################################################
if os.path.exists(train_preprocess_file):
    with open(train_preprocess_file, 'rb') as f:
        train_ids = pickle.load(f)
    converter = char_rnn.token.Converter()
    converter.load_from_file(train_converter_file)
else:
    df = pd.read_csv(train_file, sep='\t')
    df = df[df['Language'] == 'Chinese (Traditional)']
    df['len'] = df['Text'].apply(lambda x: len(str(x)))
    df = df[(df['len'] >= 60) & (df['len'] <= 200)]

    converter = char_rnn.token.Converter()
    converter.build(df['Text'])
    train_ids = converter.convert_sentences_to_ids(df['Text'])
    with open(train_preprocess_file, 'wb') as f:
        pickle.dump(train_ids, f)
    converter.save_to_file(train_converter_file)

#####################################################################
# 建立資料集
#####################################################################
dataset = char_rnn.data.Dataset(all_ids=train_ids,
                                pad_token_id=converter.pad_token_id)

#####################################################################
# 控制實驗隨機亂數
#####################################################################
torch.manual_seed(777)

#####################################################################
# 決定使用 CPU 或 GPU 進行計算
#####################################################################
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

TOTAL_STEP = 1000000
BATCH_SIZE = 256
EMBED_DIM = 100
HIDDEN_DIM = 100
LEARNING_RATE = 0.001
GRAD_CLIP_VALUE = 1

model = char_rnn.model.CharRNN(vocab_size=converter.vocab_size(),
                               embed_dim=EMBED_DIM,
                               hidden_dim=HIDDEN_DIM,
                               pad_token_id=converter.pad_token_id)

if os.path.exists(train_model_file):
    model.load_state_dict(torch.load(train_model_file))

char_rnn.model.train(model=model,
                     converter=converter,
                     dataset=dataset,
                     total_step=TOTAL_STEP,
                     batch_size=BATCH_SIZE,
                     learning_rate=LEARNING_RATE,
                     grad_clip_value=GRAD_CLIP_VALUE,
                     device=DEVICE)

torch.save(model.state_dict(), train_model_file)