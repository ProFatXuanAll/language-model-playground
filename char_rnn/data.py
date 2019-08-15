import os
import pickle
import torch
import torch.utils.data
import torch.nn.utils.rnn

#####################################################################
# 建立 data set 與 data loader
#####################################################################

class Dataset(torch.utils.data.Dataset):
    def __init__(self, all_ids, pad_token_id=0):
        self.x = [torch.LongTensor(ids[:-1]) for ids in all_ids]
        self.x_len = torch.LongTensor([len(x) for x in self.x])
        self.x = torch.nn.utils.rnn.pad_sequence(self.x,
                                                 batch_first=True,
                                                 padding_value=pad_token_id)
        self.y = [torch.LongTensor(ids[1:]) for ids in all_ids]
        self.y = torch.nn.utils.rnn.pad_sequence(self.y,
                                                 batch_first=True,
                                                 padding_value=pad_token_id)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.x_len[index], self.y[index]

    def data_loader(self, batch_size):
        return torch.utils.data.DataLoader(self,
                                           batch_size=batch_size,
                                           shuffle=True)
