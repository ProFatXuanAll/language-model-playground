import torch.utils.data
import torch.nn.utils.rnn

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        super(BaseDataset, self).__init__()

        self.text_list = kwargs.pop('text_list', [])
        self.config = kwargs.pop('config', None)
        self.tokenizer = kwargs.pop('tokenizer', None)

        self.tokenizer.build_dict(self.text_list, self.config.min_count)

        self.id_list = [torch.LongTensor(ids) for ids in self.tokenizer.encode(self.text_list)]

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        return self.id_list[index][:-1], self.id_list[index][1:]

    def collate_fn(self, batch):
        x = torch.nn.utils.rnn.pad_sequence([data[0] for data in batch],
                                            batch_first=True,
                                            padding_value=self.tokenizer.pad_token_id)
        y = torch.nn.utils.rnn.pad_sequence([data[1] for data in batch],
                                            batch_first=True,
                                            padding_value=self.tokenizer.pad_token_id)
        return x, y