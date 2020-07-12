import torch.utils.data
import torch.nn.utils.rnn
from typing import List, Tuple, Union
import lmp.tokenizer
import lmp.config


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            text_list: List[str],     
            config: lmp.config.BaseConfig,
            tokenizer: Union[lmp.tokenizer.BaseTokenizerByList, lmp.tokenizer.BaseTokenizerByDict],
            is_uncased: bool = False
    ):
        super(BaseDataset, self).__init__()

        self.text_list = text_list
        self.config = config
        self.tokenizer = tokenizer
        self.is_uncased = is_uncased  # 是否把大小寫視為不同字, default: False

        self.tokenizer.build_dict(self.text_list, self.config.min_count, self.is_uncased)
        self.id_list = [torch.LongTensor(ids) for ids in self.tokenizer.encode(self.text_list)]


    def __len__(self) -> int:
        return len(self.id_list)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.id_list[index][:-1], self.id_list[index][1:]

    def collate_fn(self, batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:

        x = torch.nn.utils.rnn.pad_sequence([data[0] for data in batch],
                                            batch_first=True,
                                            padding_value=self.tokenizer.pad_token_id)
        y = torch.nn.utils.rnn.pad_sequence([data[1] for data in batch],
                                            batch_first=True,
                                            padding_value=self.tokenizer.pad_token_id)

        return x, y
