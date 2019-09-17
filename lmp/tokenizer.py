# built-in modules
import os
import pickle
import re

class BaseTokenizer:
    def __init__(self, **kwargs):
        # padding token
        self.pad_token = kwargs.pop('pad_token', '[PAD]')
        self.pad_token_id = kwargs.pop('pad_token_id', 0)
        # class token
        self.cls_token = kwargs.pop('cls_token', '[CLS]')
        self.cls_token_id = kwargs.pop('cls_token_id', 1)
        # separation token
        self.sep_token = kwargs.pop('sep_token', '[SEP]')
        self.sep_token_id = kwargs.pop('sep_token_id', 2)
        # end-of-sentence token
        self.eos_token = kwargs.pop('eos_token', '[EOS]')
        self.eos_token_id = kwargs.pop('eos_token_id', 3)
        # unknown token
        self.unk_token = kwargs.pop('unk_token', '[UNK]')
        self.unk_token_id = kwargs.pop('unk_token_id', 4)

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

    def load_from_file(self, file_path=None):
        if file_path is None or type(file_path) != str:
            raise ValueError('argument `file_path` should be a string')
        elif not os.path.exists(file_path):
            raise FileNotFoundError('file {} does not exist'.format(file_path))

        with open(file_path, 'rb') as f:
            self.token_to_id = pickle.load(f)
        self.id_to_token = {v:i for i, v in self.token_to_id.items()}
        return self

    def save_to_file(self, file_path=None):
        if file_path is None or type(file_path) != str:
            raise ValueError('argument `file_path` should be a string')
        else:
            with open(file_path, 'wb') as f:
                pickle.dump(self.token_to_id, f)
        return self

    def tokenize(self, sentence):
        raise NotImplementedError(f'in class `{self.__class__.__name__}`: function `tokenize` not implemented yet.')

    def detokenize(self, tokens):
        raise NotImplementedError(f'in class `{self.__class__.__name__}`: function `detokenize` not implemented yet.')

    def convert_sentences_to_tokens(self, all_sentences):
        return [self.tokenize(sentence) for sentence in all_sentences]

    def convert_tokens_to_sentences(self, all_tokens):
        return [self.detokenize(tokens) for tokens in all_tokens]

    def convert_tokens_to_ids(self, all_tokens):
        result = []
        for tokens in all_tokens:
            ids = []
            for token in tokens:
                if token in self.token_to_id:
                    ids.append(self.token_to_id[token])
                else:
                    ids.append(self.unk_token_id)
            result.append(ids)
        return result

    def convert_ids_to_tokens(self, all_ids):
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

    def convert_sentences_to_ids(self, all_sentences):
        return self.convert_tokens_to_ids(self.convert_sentences_to_tokens(all_sentences))

    def convert_ids_to_sentences(self, all_ids):
        return self.convert_tokens_to_sentences(self.convert_ids_to_tokens(all_ids))

    def encode(self, all_sentences):
        all_ids = self.convert_sentences_to_ids(all_sentences)

        for id_list in all_ids:
            id_list.append(self.eos_token_id)

        return all_ids

    def decode(self, all_ids):
        all_sentences = self.convert_ids_to_sentences(all_ids)

        result = []
        pattern = re.escape(self.eos_token)
        for sentence in all_sentences:
            result.append(re.sub(pattern, '', sentence))

        return result

    def build_dict(self, all_sentences, min_count=0):
        all_sentences = self.convert_sentences_to_tokens(all_sentences)
        token_counter = {}

        for sentence in all_sentences:
            for token in sentence:
                if token not in token_counter:
                    token_counter[token] = 0
                token_counter[token] += 1

        index = len(self.token_to_id)

        for token, count in token_counter.items():
            if count > min_count:
                self.token_to_id[token] = index
                self.id_to_token[index] = token
                index += 1

        return self

    def vocab_size(self):
        return len(self.token_to_id)

class CharTokenizer(BaseTokenizer):
    def __init__(self, **kwargs):
        super(CharTokenizer, self).__init__(**kwargs)

    def tokenize(self, sentence):
        return list(sentence)

    def detokenize(self, tokens):
        return ''.join(tokens)

class WhiteSpaceTokenizer(BaseTokenizer):
    def __init__(self, **kwargs):
        super(WhiteSpaceTokenizer, self).__init__(**kwargs)

    def tokenize(self, sentence):
        return re.split(r'\s+', sentence)

    def detokenize(self, tokens):
        return ' '.join(tokens)