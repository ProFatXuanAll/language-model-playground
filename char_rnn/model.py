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


#############################################
# 訓練
#############################################
def train(model=None,
          converter=None,
          dataset=None,
          total_step=100000,
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
                                          reduction='none')
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

            loss.backward(torch.ones_like(loss))
            torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip_value)
            optimizer.step()

            step += 1
            if step % 1000 == 0:
                print('step: {}/{}, loss: {}'.format(step,
                                                     total_step,
                                                     float(loss.mean())))

            if step >= total_step:
                break


#############################################
# 生成
#############################################
def generator(model=None,
              converter=None,
              begin_of_sentence='',
              beam_width=4,
              max_len=200):

    if begin_of_sentence is None or len(begin_of_sentence) == 0:
        raise ValueError('`begin_of_sentence` should be list type object.')

    generate_result = []

    with torch.no_grad():
        all_ids = converter.convert_sentences_to_ids([begin_of_sentence])
        all_ids_prob = [0]

        while True:
            active_ids = []
            active_ids_prob = []

            # 決定當前還有哪些句子需要進行生成
            for i in range(len(all_ids)):
                if all_ids[i][-1] != converter.eos_token_id and len(all_ids[i]) < max_len:
                    active_ids.append(all_ids[i])
                    active_ids_prob.append(all_ids_prob[i])
                elif len(generate_result) < beam_width:
                    generate_result.append(all_ids[i])

            # 如果沒有需要生成的句子就結束迴圈
            if not active_ids or len(generate_result) >= beam_width:
                break

            batch_x = [torch.LongTensor(ids) for ids in active_ids]
            batch_x = torch.nn.utils.rnn.pad_sequence(batch_x,
                                                      batch_first=True,
                                                      padding_value=converter.pad_token_id)
            batch_x_len = torch.LongTensor([len(ids) for ids in active_ids])

            batch_pred_y = model(batch_x, batch_x_len)

            # 從各個句子中的 beam 挑出前 beam_width 個最大值
            top_value = {}
            for beam_id in range(len(active_ids)):
                current_beam_vocab_pred = batch_pred_y[beam_id][len(active_ids[beam_id]) - 1]
                current_beam_vocab_pred = torch.nn.functional.softmax(current_beam_vocab_pred, dim=0)

                current_beam_top_value = [{
                    'vocab_id': converter.eos_token_id,
                    'value': 0
                } for i in range(beam_width)]

                for vocab_id in range(converter.vocab_size()):
                    if current_beam_vocab_pred[vocab_id] < current_beam_top_value[0]['value']:
                        continue

                    for level in range(beam_width):
                        if current_beam_vocab_pred[vocab_id] < current_beam_top_value[level]['value']:
                            level -= 1
                            break

                    for tmp in range(level):
                        current_beam_top_value[tmp] = current_beam_top_value[tmp + 1]

                    current_beam_top_value[level] = {'vocab_id': vocab_id,
                                                     'value': current_beam_vocab_pred[vocab_id]}

                top_value[beam_id] = current_beam_top_value

            # print('top value')
            # print(top_value)
            # 從 beam_width ** 2 中挑出 beam_width 個最大值
            final_top_value = []

            for i in range(beam_width):
                max_value_beam_id = 0
                max_value_vocab_id = converter.eos_token_id
                min_value = 999999999

                for beam_id in range(len(top_value)):
                    value = -torch.log(top_value[beam_id][-1]['value']) + active_ids_prob[beam_id]
                    if value < min_value:
                        max_value_beam_id = beam_id
                        min_value = value
                        max_value_vocab_id = top_value[beam_id][-1]['vocab_id']

                final_top_value.append({
                    'beam_id': max_value_beam_id,
                    'vocab_id': max_value_vocab_id,
                    'value': min_value
                })

                top_value[max_value_beam_id].pop()

            # print('final')
            # print(final_top_value)
            # back to all_ids
            all_ids = []
            all_ids_prob = []
            for obj in final_top_value:
                all_ids.append(active_ids[obj['beam_id']] + [obj['vocab_id']])
                all_ids_prob.append(obj['value'])
            # print('all ids')
            # print(all_ids)
            # print()

    for ids in generate_result:
        if ids[-1] == converter.eos_token_id:
            ids.pop()
    return converter.convert_ids_to_sentences(generate_result)

