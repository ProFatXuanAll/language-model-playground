r"""preprocessing training dataset.

Usage:
    dataset = lmp.dataset.BaseDataset(...)
"""

import torch.utils.data
import torch.nn.utils.rnn
from typing import List, Tuple, Union
import lmp.tokenizer
import lmp.config


class BaseDataset(torch.utils.data.Dataset):
    r"""Used to preprocess the data.

    Attributes:
        text_list:
            All sentences of dataset.
            Be Used to build vocabulary dict.
        config:
            Configuration for vocabulary dict.
            Come from lmp.config.BaseConfig.
        tokenizer:
            Encode sentences to ids.
        id_list:
            Save ids of encoding sentences result.

    """

    def __init__(
            self,
            text_list: List[str],
            config: lmp.config.BaseConfig,
            tokenizer: Union[lmp.tokenizer.BaseTokenizerByList,
                             lmp.tokenizer.BaseTokenizerByDict]
    ):
        super(BaseDataset, self).__init__()

        self.text_list = text_list
        self.config = config
        self.tokenizer = tokenizer

        self.tokenizer.build_dict(
            self.text_list, self.config.min_count, self.config.is_uncased)
        self.id_list = [torch.LongTensor(
            ids) for ids in self.tokenizer.encode(self.text_list)]

    def __len__(self) -> int:
        return len(self.id_list)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.id_list[index][:-1], self.id_list[index][1:]

    def collate_fn(self, batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Processing training data, make each shape of x,y fit [batch_size , Max_seq_len].
        Max_seq_len means maximum of sentence's length in each batch. 

        """
        x = torch.nn.utils.rnn.pad_sequence([data[0] for data in batch],
                                            batch_first=True,
                                            padding_value=self.tokenizer.pad_token_id)
        y = torch.nn.utils.rnn.pad_sequence([data[1] for data in batch],
                                            batch_first=True,
                                            padding_value=self.tokenizer.pad_token_id)

        return x, y
