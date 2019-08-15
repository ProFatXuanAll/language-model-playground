import os
import pickle
import torch.utils.data

#####################################################################
# 轉換句子成為 token 且建立 id
#####################################################################
class Converter(object):
    def __init__(self,
                 pad_token='[PAD]', pad_token_id=0,
                 cls_token='[CLS]', cls_token_id=1,
                 sep_token='[SEP]', sep_token_id=2,
                 eos_token='[EOS]', eos_token_id=3,
                 unk_token='[UNK]', unk_token_id=4):

        self.pad_token = pad_token
        self.pad_token_id = pad_token_id
        self.cls_token = cls_token
        self.cls_token_id = cls_token_id
        self.sep_token = sep_token
        self.sep_token_id = sep_token_id
        self.eos_token = eos_token
        self.eos_token_id = eos_token_id
        self.unk_token = unk_token
        self.unk_token_id = unk_token_id

        self.token_to_id = {}
        self.id_to_token = {}

        self.token_to_id[pad_token] = pad_token_id
        self.id_to_token[pad_token_id] = pad_token
        self.token_to_id[cls_token] = cls_token_id
        self.id_to_token[cls_token_id] = cls_token
        self.token_to_id[sep_token] = sep_token_id
        self.id_to_token[sep_token_id] = sep_token
        self.token_to_id[eos_token] = eos_token_id
        self.id_to_token[eos_token_id] = eos_token
        self.token_to_id[unk_token] = unk_token_id
        self.id_to_token[unk_token_id] = unk_token

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

    def tokenizer(self, sentence):
        return list(sentence)

    def detokenizer(self, tokens):
        return ''.join(tokens)

    def convert_sentences_to_tokens(self, all_sentences):
        return [self.tokenizer(sentence) for sentence in all_sentences]

    def convert_tokens_to_sentences(self, all_tokens):
        return [self.detokenizer(tokens) for tokens in all_tokens]

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

    def build(self, all_sentences):
        index = len(self.token_to_id)

        for tokens in self.convert_sentences_to_tokens(all_sentences):
            for token in tokens:
                if token not in self.token_to_id:
                    self.token_to_id[token] = index
                    self.id_to_token[index] = token
                    index += 1
        return self

    def vocab_size(self):
        return len(self.token_to_id)