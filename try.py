import torch
import torch.nn

class RNNModel(torch.nn.Module):
    def __init__(
        self,
        d_emb: int,
        d_hid: int,
        dropout: float,
        num_linear_layers: int,
        num_rnn_layers: int,
        pad_token_id: int,
        vocab_size: int
    ):
        super().__init__()

        self.emb_layer = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_emb,
            padding_idx=pad_token_id
        )
        self.emb_dropout = torch.nn.Dropout(dropout)

        self.proj_emb_to_hid = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=d_emb,
                out_features=d_hid
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )

        self.rnn_layer = torch.nn.RNN(
            input_size=d_hid,
            hidden_size=d_hid,
            num_layers=num_rnn_layers,
            dropout=dropout,
            batch_first=True
        )

        # Sequential linear layer(s).
        # Dimension: (H, H).
        proj_hid_to_emb = []
        for _ in range(num_linear_layers - 1):
            proj_hid_to_emb.append(torch.nn.Dropout(dropout))
            proj_hid_to_emb.append(
                torch.nn.Linear(
                    in_features=d_hid,
                    out_features=d_hid
                )
            )
            proj_hid_to_emb.append(torch.nn.ReLU())

        # Sequential linear layer(s)' last layer.
        # Dimension: (H, E).
        proj_hid_to_emb.append(torch.nn.Dropout(dropout))
        proj_hid_to_emb.append(
            torch.nn.Linear(
                in_features=d_hid,
                out_features=d_emb
            )
        )
        proj_hid_to_emb.append(torch.nn.ReLU())
        proj_hid_to_emb.append(torch.nn.Dropout(dropout))
        self.proj_hid_to_emb = torch.nn.Sequential(*proj_hid_to_emb)

        self.query = torch.nn.Linear(
            in_features=d_hid,
            out_features=d_hid
        )
        self.key = torch.nn.Linear(
            in_features=d_hid,
            out_features=d_hid
        )

    def forward(self, x):
        x = self.emb_dropout(self.emb_layer(x))

        ht = self.proj_emb_to_hid(x)

        ht, tmp = self.rnn_layer(ht)

        q = self.query(ht)  # (B, S, H)
        k = self.key(ht)  # (B, S, H)
        e_matrix = q.matmul(k.transpose(1,2)) # (B, S, H) * (B, H, S) = (B, S, S)
        mask = torch.ones(e_matrix.size()[1:], dtype=torch.int8)
        mask = torch.tril(mask)
        mask = mask.unsqueeze(0).repeat(e_matrix.size(0), 1, 1)
        e_matrix = e_matrix.mul(mask)
        print(e_matrix.size())
        a_matrix = torch.nn.functional.softmax(e_matrix, dim=2) # (B, S, S)
        print('a_matrix:', a_matrix.size())
        print('ht.transpose(1, 2):', ht.size())
        ht = a_matrix.matmul(ht)







        ht = self.proj_hid_to_emb(ht)

        return ht.matmul(self.emb_layer.weight.transpose(0, 1))


batch_size = 32
seq_len = 20

d_emb = 10
d_hid = 10
dropout = 0.1
num_linear_layers = 1
num_rnn_layers = 1
pad_token_id = 0
vocab_size = 100

model = RNNModel(
    d_emb=d_emb,
    d_hid=d_hid,
    dropout=dropout,
    num_linear_layers=num_linear_layers,
    num_rnn_layers=num_rnn_layers,
    pad_token_id=pad_token_id,
    vocab_size=vocab_size
)

x = torch.randint(low=0, high=99, size=(batch_size, seq_len))
print('x的size: ', x.shape)
print('\n要進入 model: ')
y = model(x)
print('出來 model:\n')

print('y的size: ', y.shape)

python run_train.py --experiment att_gru_3 --batch_size 64 --checkpoint -1 --checkpoint_step 500 --d_emb 300 --d_hid 600 --dataset news_collection_title --dropout 0.1 --epoch 30 --is_uncased --learning_rate 1e-5 --max_norm 1.0 --max_seq_len 60 --min_count 1 --model_class att_gru --num_linear_layers 1 --num_rnn_layers 1 --optimizer_class adam --seed 42 --tokenizer_class char_dict

python run_train.py --experiment att_lstm_1 --checkpoint 89200 --epoch 100


python run_train.py --experiment att_gru_2 --checkpoint 28500 --epoch 100

python run_generate.py --experiment 1 --checkpoint 223000 --begin_of_sequence 今天 --beam_width 4 --max_seq_len 60

python run_generate.py --experiment 2 --checkpoint 66900 --begin_of_sequence vu/八課 --beam_width 4 --max_seq_len 60

python run_generate.py --experiment att_lstm_1 --checkpoint 223000 --begin_of_sequence 韓國 --beam_width 4 --max_seq_len 60

CUDA_VISIBLE_DEVICES=1

CUDA_VISIBLE_DEVICES=1 python run_train.py --experiment att_res_gru_9 --batch_size 128 --checkpoint -1 --checkpoint_step 500 --d_emb 300 --d_hid 600 --dataset news_collection_desc --dropout 0.05 --epoch 40 --is_uncased --learning_rate 1e-3 --max_norm 5.0 --max_seq_len 100 --min_count 1 --model_class att_res_rnn --num_linear_layers 2 --num_rnn_layers 2 --optimizer_class adam --seed 42 --tokenizer_class char_dict

CUDA_VISIBLE_DEVICES=1 python run_train.py --experiment att_res_gru_6 --checkpoint 64320 --epoch 100

CUDA_VISIBLE_DEVICES=1 python run_generate.py --experiment att_gru_2 --checkpoint 28500 --begin_of_sequence 總統 --beam_width 4 --max_seq_len 60
CUDA_VISIBLE_DEVICES=1 python run_generate.py --experiment att_gru_3 --checkpoint 66900 --begin_of_sequence 總統 --beam_width 4 --max_seq_len 60
CUDA_VISIBLE_DEVICES=1 python run_generate.py --experiment att_gru_4 --checkpoint 64320 --begin_of_sequence 總統 --beam_width 4 --max_seq_len 60
CUDA_VISIBLE_DEVICES=1 python run_generate.py --experiment att_gru_5 --checkpoint 128640 --begin_of_sequence 總統 --beam_width 4 --max_seq_len 60
CUDA_VISIBLE_DEVICES=1 python run_generate.py --experiment att_lstm_1 --checkpoint 223000 --begin_of_sequence 總統 --beam_width 4 --max_seq_len 60

CUDA_VISIBLE_DEVICES=1 python run_generate.py --experiment att_res_gru_6 --checkpoint 214400 --begin_of_sequence 台灣總統 --beam_width 2 --max_seq_len 60
CUDA_VISIBLE_DEVICES=1 python run_generate.py --experiment att_res_rnn_1 --checkpoint 13500 --begin_of_sequence 中共人民解放軍 --beam_width 2 --max_seq_len 60


CUDA_VISIBLE_DEVICES=1 python run_generate.py --experiment att_res_gru_7 --checkpoint 42880 --begin_of_sequence 中共人民 --beam_width 2 --max_seq_len 60
CUDA_VISIBLE_DEVICES=1 python run_generate.py --experiment rnn --checkpoint 4000 --begin_of_sequence 中共人民 --beam_width 2 --max_seq_len 60
