r"""tokenizing all sentences of dataset,
    encoding tokens to ids,
    decoding ids to token.

Usage:
    tokenizer = lmp.tokenizer.CharTokenizerByList(...)
    tokenizer = lmp.tokenizer.CharTokenizerByDict(...)

"""

# built-in modules
import os
import pickle
import re
import abc
from typing import List, Dict

# type(token_to_id) is list (in order to save memory space)


class BaseTokenizerByList:
    r"""Define how to build dictionry, encode tokens to id and decode ids back to token.
    Using list structure to implement token_to_id.

    Attributes:
        pad_token:
            Used for pad sequence to same length.
            All sequences in batch should have the same length.
        cls_token:
            Stand for classification.
            For the sentence classification task.
        sep_token:
            Stand for separating sentences.
            For the next sentence prediction task.
        eos_token:
            Stand for End of sentence token.
            As soon as decoder generates this token we consider the answer to be complete.
        unk_token:
            Stand for Unknown token.
            Used to replace the words that did not show in our vocabulary.

        pad_token_id:
            This id stand for pad token.
        cls_token_id:
            This id stand for cls token.
        sep_token_id:
            This id stand for sep token.
        eos_token_id:
            This id stand for eos token.
        unk_token_id:
            This id stand for unk token.

        token_to_id:
            Convert token to id by using token_to_id.index(token).
            Convert id back to token by using token_to_id[id]. 
            using list structure to save memory space.

    """

    def __init__(
            self,
            pad_token: str = '[PAD]', pad_token_id: int = 0,
            cls_token: str = '[CLS]', cls_token_id: int = 1,
            sep_token: str = '[SEP]', sep_token_id: int = 2,
            eos_token: str = '[EOS]', eos_token_id: int = 3,
            unk_token: str = '[UNK]', unk_token_id: int = 4
    ):

        # padding token
        self.pad_token = pad_token
        self.pad_token_id = pad_token_id
        # class token
        self.cls_token = cls_token
        self.cls_token_id = cls_token_id
        # separation token
        self.sep_token = sep_token
        self.sep_token_id = sep_token_id
        # end-of-sentence token
        self.eos_token = eos_token
        self.eos_token_id = eos_token_id
        # unknown token
        self.unk_token = unk_token
        self.unk_token_id = unk_token_id

        self.token_to_id: List[str] = []  # 用 list 節省記憶體

        self.token_to_id.append(self.pad_token)
        self.token_to_id.append(self.cls_token)
        self.token_to_id.append(self.sep_token)
        self.token_to_id.append(self.eos_token)
        self.token_to_id.append(self.unk_token)

    @classmethod
    def load_from_file(cls, file_path: str = None):
        self = cls()

        if file_path is None or type(file_path) != str:
            raise ValueError('argument `file_path` should be a string')
        elif not os.path.exists(file_path):
            raise FileNotFoundError('file {} does not exist'.format(file_path))

        with open(file_path, 'rb') as f:
            self.token_to_id = pickle.load(f)

        return self

    def save_to_file(self, file_path: str = None):
        if file_path is None or type(file_path) != str:
            raise ValueError('argument `file_path` should be a string')
        else:
            with open(file_path, 'wb') as f:
                pickle.dump(self.token_to_id, f)
        return self

    @abc.abstractmethod
    def tokenize(self, sentence: str):
        raise NotImplementedError(
            f'in class `{self.__class__.__name__}`: function `tokenize` not implemented yet.')

    @abc.abstractmethod
    def detokenize(self, tokens: List):
        raise NotImplementedError(
            f'in class `{self.__class__.__name__}`: function `detokenize` not implemented yet.')

    def convert_sentences_to_tokens(self, all_sentences: List[str]) -> List[List[str]]:
        return [self.tokenize(sentence) for sentence in all_sentences]

    def convert_tokens_to_sentences(self, all_tokens: List[List[str]]) -> List[str]:
        return [self.detokenize(tokens) for tokens in all_tokens]

    def convert_tokens_to_ids(self, all_tokens: List[List[str]]) -> List[List[int]]:
        result = []
        for tokens in all_tokens:
            ids = []
            for token in tokens:
                if token in self.token_to_id:
                    # 取出該 token 在 list 的 index
                    ids.append(self.token_to_id.index(token))
                else:
                    ids.append(self.unk_token_id)
            result.append(ids)

        return result

    def convert_ids_to_tokens(self, all_ids: List[List[int]]) -> List[List[str]]:
        result = []
        for ids in all_ids:
            tokens = []
            for index in ids:
                if index <= len(self.token_to_id):
                    tokens.append(self.token_to_id[index])
                else:
                    tokens.append(self.unk_token)
            result.append(tokens)

        return result

    def convert_sentences_to_ids(self, all_sentences: List[str]) -> List[List[int]]:
        return self.convert_tokens_to_ids(self.convert_sentences_to_tokens(all_sentences))

    def convert_ids_to_sentences(self, all_ids: List[List[int]]) -> List[str]:
        return self.convert_tokens_to_sentences(self.convert_ids_to_tokens(all_ids))

    def encode(self, all_sentences: List[str]) -> List[List[int]]:
        all_ids = self.convert_sentences_to_ids(all_sentences)

        for id_list in all_ids:
            id_list.append(self.eos_token_id)

        return all_ids

    def decode(self, all_ids: List[List[int]]) -> List[str]:
        all_sentences = self.convert_ids_to_sentences(all_ids)

        result = []
        pattern = re.escape(self.eos_token)
        for sentence in all_sentences:
            result.append(re.sub(pattern, '', sentence))

        return result

    def build_dict(self, all_sentences: List[str], min_count: int = 0, is_uncased: bool = False):
        """Build a vocabulary list of all tokens, dict is sorted by token frenquence(descending order).

        Args:
            min_count: 
                Minimum of token'sfrequence.
                if token's frequence is larger than min_count, then add token to token_to_id
            is_uncased: 
                Determine if convert all upper case into lower case.
        """
        if is_uncased:
            all_sentences = [text.lower() for text in all_sentences]

        all_tokens = self.convert_sentences_to_tokens(all_sentences)
        token_counter = {}

        for sentence in all_tokens:
            for token in sentence:
                if token not in token_counter:
                    token_counter[token] = 0
                token_counter[token] += 1

        # 按照詞頻排序，由高到低
        token_counter = dict(sorted(token_counter.items(),
                                    key=lambda x: x[1], reverse=True))

        for token, count in token_counter.items():
            if count > min_count:
                self.token_to_id.append(token)

        return self

    def vocab_size(self) -> int:
        return len(self.token_to_id)


# type(token_to_id) is dictionary
class BaseTokenizerByDict:
    r"""Define how to build dictionry, encode tokens to id and decode ids back to token.
    Using list structure to implement token_to_id.

    Attributes:
        pad_token:
            Used for pad sequence to same length.
            All sequences in batch should have the same length.
        cls_token:
            Stand for classification.
            For the sentence classification task.
        sep_token:
            Stand for separating sentences.
            For the next sentence prediction task.
        eos_token:
            Stand for End of sentence token.
            As soon as decoder generates this token we consider the answer to be complete.
        unk_token:
            Stand for Unknown token.
            Used to replace the words that did not show in our vocabulary.

        pad_token_id:
            This id stand for pad token.
        cls_token_id:
            This id stand for cls token.
        sep_token_id:
            This id stand for sep token.
        eos_token_id:
            This id stand for eos token.
        unk_token_id:
            This id stand for unk token.

        token_to_id:
            Convert token to id.
        id_to_token:
            Convert id back to token.
    """

    def __init__(
            self,
            pad_token: str = '[PAD]', pad_token_id: int = 0,
            cls_token: str = '[CLS]', cls_token_id: int = 1,
            sep_token: str = '[SEP]', sep_token_id: int = 2,
            eos_token: str = '[EOS]', eos_token_id: int = 3,
            unk_token: str = '[UNK]', unk_token_id: int = 4
    ):

        # padding token
        self.pad_token = pad_token
        self.pad_token_id = pad_token_id
        # class token
        self.cls_token = cls_token
        self.cls_token_id = cls_token_id
        # separation token
        self.sep_token = sep_token
        self.sep_token_id = sep_token_id
        # end-of-sentence token
        self.eos_token = eos_token
        self.eos_token_id = eos_token_id
        # unknown token
        self.unk_token = unk_token
        self.unk_token_id = unk_token_id

        self.token_to_id = {}
        self.id_to_token = {}

        self.token_to_id[self.pad_token] = self.pad_token_id
        self.id_to_token[self.pad_token_id] = self.pad_token
        self.token_to_id[self.cls_token] = self.cls_token_id
        self.id_to_token[self.cls_token_id] = self.cls_token
        self.token_to_id[self.sep_token] = self.sep_token_id
        self.id_to_token[self.sep_token_id] = self.sep_token
        self.token_to_id[self.eos_token] = self.eos_token_id
        self.id_to_token[self.eos_token_id] = self.eos_token
        self.token_to_id[self.unk_token] = self.unk_token_id
        self.id_to_token[self.unk_token_id] = self.unk_token

    @classmethod
    def load_from_file(cls, file_path: str = None):
        self = cls()

        if file_path is None or type(file_path) != str:
            raise ValueError('argument `file_path` should be a string')
        elif not os.path.exists(file_path):
            raise FileNotFoundError('file {} does not exist'.format(file_path))

        with open(file_path, 'rb') as f:
            self.token_to_id = pickle.load(f)
        self.id_to_token = {v: i for i, v in self.token_to_id.items()}
        return self

    def save_to_file(self, file_path: str = None):
        if file_path is None or type(file_path) != str:
            raise ValueError('argument `file_path` should be a string')
        else:
            with open(file_path, 'wb') as f:
                pickle.dump(self.token_to_id, f)
        return self

    @abc.abstractmethod
    def tokenize(self, sentence: str):
        raise NotImplementedError(
            f'in class `{self.__class__.__name__}`: function `tokenize` not implemented yet.')

    @abc.abstractmethod
    def detokenize(self, tokens: List):
        raise NotImplementedError(
            f'in class `{self.__class__.__name__}`: function `detokenize` not implemented yet.')

    def convert_sentences_to_tokens(self, all_sentences: List[str]) -> List[List[str]]:
        return [self.tokenize(sentence) for sentence in all_sentences]

    def convert_tokens_to_sentences(self, all_tokens: List[List[str]]) -> List[str]:
        return [self.detokenize(tokens) for tokens in all_tokens]

    def convert_tokens_to_ids(self, all_tokens: List[List[str]]) -> List[List[int]]:
        result = []
        for tokens in all_tokens:
            ids = []
            for token in tokens:
                if token in self.token_to_id:
                    ids.append(self.token_to_id[token])  # 取出該token在 list的index
                else:
                    ids.append(self.unk_token_id)
            result.append(ids)

        return result

    def convert_ids_to_tokens(self, all_ids: List[List[int]]) -> List[List[str]]:
        result = []
        for ids in all_ids:
            tokens = []
            for index in ids:
                if index in self.id_to_token:
                    tokens.append(self.id_to_token[index])
                else:
                    tokens.append(self.unk_token)
            result.append(tokens)

        return result

    def convert_sentences_to_ids(self, all_sentences: List[str]) -> List[List[int]]:
        return self.convert_tokens_to_ids(self.convert_sentences_to_tokens(all_sentences))

    def convert_ids_to_sentences(self, all_ids: List[List[int]]) -> List[str]:
        return self.convert_tokens_to_sentences(self.convert_ids_to_tokens(all_ids))

    def encode(self, all_sentences: List[str]) -> List[List[int]]:
        all_ids = self.convert_sentences_to_ids(all_sentences)

        for id_list in all_ids:
            id_list.append(self.eos_token_id)

        return all_ids

    def decode(self, all_ids: List[List[int]]) -> List[str]:
        all_sentences = self.convert_ids_to_sentences(all_ids)

        result = []
        pattern = re.escape(self.eos_token)
        for sentence in all_sentences:
            result.append(re.sub(pattern, '', sentence))

        return result

    def build_dict(self, all_sentences: List[str], min_count: int = 0, is_uncased: bool = False):
        """Build a vocabulary dict of all tokens, dict is sorted by token frenquence(descending order).

        Args:
            min_count:
                Minimum of token'sfrequence.
                if token's frequence is larger than min_count, then add token to token_to_id
            is_uncased:
                Determine if convert all upper case into lower case.

        """
        if is_uncased:
            all_sentences = [text.lower() for text in all_sentences]

        all_tokens = self.convert_sentences_to_tokens(all_sentences)
        token_counter = {}

        for sentence in all_tokens:
            for token in sentence:
                if token not in token_counter:
                    token_counter[token] = 0
                token_counter[token] += 1

        # 按照詞頻排序，由高到低
        token_counter = dict(sorted(token_counter.items(),
                                    key=lambda x: x[1], reverse=True))

        index = len(self.token_to_id)

        for token, count in token_counter.items():
            if count > min_count:
                self.token_to_id[token] = index
                self.id_to_token[index] = token
                index += 1

        return self

    def vocab_size(self) -> int:
        return len(self.token_to_id)


class CharTokenizerByList(BaseTokenizerByList):
    r"""Tokenizing sentence by spliting all characters.
    """

    def __init__(self, **kwargs):
        super(CharTokenizerByList, self).__init__(**kwargs)

    def tokenize(self, sentence: str) -> List[str]:
        return list(sentence)

    def detokenize(self, tokens: List) -> str:
        return ''.join(tokens)


class CharTokenizerByDict(BaseTokenizerByDict):
    r"""Tokenizing sentence by spliting all characters.
    """

    def __init__(self, **kwargs):
        super(CharTokenizerByDict, self).__init__(**kwargs)

    def tokenize(self, sentence: str) -> List[str]:
        return list(sentence)

    def detokenize(self, tokens: List) -> str:
        return ''.join(tokens)


class WhiteSpaceTokenizer(BaseTokenizerByList):
    r"""Tokenizing sentence by spliting spaces.
    """

    def __init__(self, **kwargs):
        super(WhiteSpaceTokenizer, self).__init__(**kwargs)

    def tokenize(self, sentence: str) -> List[str]:
        return re.split(r'\s+', sentence)

    def detokenize(self, tokens: List) -> str:
        return ' '.join(tokens)
