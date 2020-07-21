r"""preprocessing training dataset.

Usage:
    dataset = lmp.dataset.BaseDataset(...)
"""
import torch.utils.data
import torch.nn.utils.rnn
from typing import List, Tuple, Union, Callable
import lmp.tokenizer
import pickle
import lmp
import os

###############################################################################
# Define types for type annotation.
###############################################################################
CollateFnReturn = Tuple[
    torch.LongTensor,
    torch.LongTensor
]


class BaseDataset(torch.utils.data.Dataset):
    r"""Used to preprocess the data.

    Attributes:
        text_list:
            All sentences of dataset.
            Be Used to build vocabulary dict.
    """

    def __init__(
            self,
            text_list: List[str]
    ):
        super(BaseDataset, self).__init__()
        self.text_list = text_list

    def __len__(self) -> int:
        return len(self.text_list)

    # Tuple[torch.Tensor, torch.Tensor]
    def __getitem__(self, index: int) -> str:
        return self.text_list[index]

    @staticmethod
    def creat_collate_fn(
            tokenizer:  Union[lmp.tokenizer.BaseTokenizerByList,
                              lmp.tokenizer.BaseTokenizerByDict],
            max_seq_len: int = -1
    ):
        r"""
        Processing training data, make each shape of x,y fit [batch_size , Max_seq_len].
        Max_seq_len means maximum of sentence's length in each batch.

        Attributes:
            tokenizer:
                Encode sentences to ids.
            max_seq_len:
                Indicate each sentence's max length.
        """

        def collate_fn(batch: List['str']) -> CollateFnReturn:
            batch_id_list = torch.LongTensor(
                tokenizer.encode(batch, max_seq_len))

            x = batch_id_list[:, :-1]

            y = batch_id_list[:, 1:]

            y = y.contiguous()

            return x, y

        return collate_fn
