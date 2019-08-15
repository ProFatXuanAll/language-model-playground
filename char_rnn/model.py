import torch
import torch.nn
import torch.utils.data
import torch.nn.utils.rnn

############################################################
# 建立模型
############################################################

class CharRNN(torch.nn.Module):
    def __init__(self, vocab_size,
                       embed_dim,
                       hidden_dim,
                       pad_token_id=0):
        super(CharRNN, self).__init__()

        #########################################
        # 先做 character embedding
        #########################################
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=embed_dim,
                                            padding_idx=pad_token_id)

        self.embedding.weight.data.normal_(mean=0, std=0.5)


        #########################################
        # 丟進兩層 LSTM
        #########################################
        self.rnn_layer1 = torch.nn.LSTM(input_size=embed_dim,
                                        hidden_size=hidden_dim,
                                        batch_first=True)

        self.rnn_layer2 = torch.nn.LSTM(input_size=hidden_dim,
                                        hidden_size=hidden_dim,
                                        batch_first=True)


    def forward(self, batch_x, batch_x_lens):
        ######################################################################
        # 維度: (Batch_Size, Sequence_Length)
        ######################################################################
        batch_x = self.embedding(batch_x)
        ######################################################################
        # 維度: (Batch_Size, Sequence_Length, Embedding_Dimension)
        ######################################################################
        # batch_x = torch.nn.utils.rnn.pack_padded_sequence(batch_x,
        #                                                   batch_x_lens,
        #                                                   batch_first=True,
        #                                                   enforce_sorted=False)

        batch_x, _ = self.rnn_layer1(batch_x)
        ######################################################################
        # 維度: (Batch_Size, Sequence_Length, Hidden_Dimension)
        ######################################################################
        batch_x, _ = self.rnn_layer2(batch_x)

        # batch_x, _ = torch.nn.utils.rnn.pad_packed_sequence(batch_x,
        #                                                     batch_first=True)

        ######################################################################
        # 維度: (Batch_Size, Sequence_Length, Hidden_Dimension)
        ######################################################################
        batch_x = batch_x.matmul(self.embedding.weight.transpose(0, 1))

        ######################################################################
        # 維度: (Batch_Size, Sequence_Length, Vocabulary_Size)
        ######################################################################
        return batch_x

def train(model=None,
          converter=None,
          dataset=None,
          total_step=10000,
          batch_size=32,
          learning_rate=0.001,
          grad_clip_value=1,
          device=torch.device('cpu')):
    #############################################
    # 選擇 Loss Function 與 梯度更新演算法
    #############################################
    vocab_size = converter.vocab_size()
    pad_token_id = converter.pad_token_id

    data_loader = dataset.data_loader(batch_size=batch_size)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id,
                                          reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate)

    model = model.to(device)
    model.train()

    step = 0
    while step < total_step:
        for batch_x, batch_x_lens, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_x_lens = batch_x_lens.to(device)
            batch_y = batch_y.to(device).view(-1)
            optimizer.zero_grad()

            batch_pred_y = model(batch_x, batch_x_lens)
            batch_pred_y = batch_pred_y.view(-1, vocab_size)

            loss = criterion(batch_pred_y, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip_value)
            optimizer.step()

            step += 1
            if step % 1000 == 0:
                print('step: {}/{}, loss: {}'.format(step,
                                                     total_step,
                                                     float(loss)))

            if step >= total_step:
                break


# ########
# # 生成
# ########
# def generator(start_char, max_len=200):

#     char_list = [char_to_id[start_char]]

#     next_char = None

#     while len(char_list) < max_len:
#         x = torch.LongTensor(char_list).unsqueeze(0)
#         x = self.embedding(x)
#         _, (ht, _) = self.rnn_layer1(x)
#         _, (ht, _) = self.rnn_layer2(ht)
#         y = ht.matmul(self.embedding.weight.transpose(0, 1))

#         next_char = np.argmax(y.numpy())

#         if next_char == char_to_id['<eos>']:
#             break

#         char_list.append(next_char)

#     return [id_to_char[ch_id] for ch_id in char_list]

# #############################################
# # 訓練
# #############################################


# #############################################
# # 測試生成
# #############################################
# with torch.no_grad():
#     model = model.cpu()
#     print(model.generator('網'))
#     print(model.generator('地'))
