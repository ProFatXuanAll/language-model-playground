# built-in modules
import os
import pickle

class BaseConfig:
    def __init__(self, **kwargs):
        self.batch_size = kwargs.pop('batch_size', 1)
        self.dropout = kwargs.pop('dropout', 0)
        self.embedding_dim = kwargs.pop('embedding_dim', 1)
        self.epoch = kwargs.pop('epoch', 1)
        self.grad_clip_value = kwargs.pop('grad_clip_value', 1)
        self.hidden_dim = kwargs.pop('hidden_dim', 1)
        self.learning_rate = kwargs.pop('learning_rate', 10e-4)
        self.min_count = kwargs.pop('min_count', 0)
        self.num_rnn_layers = kwargs.pop('num_rnn_layers', 1)
        self.seed = kwargs.pop('seed', 1)

    @classmethod
    def load_from_file(cls, file_path=None, **kwargs):
        self = cls()

        if file_path is None or type(file_path) != str:
            raise ValueError('argument `file_path` should be a string')
        elif not os.path.exists(file_path):
            raise FileNotFoundError('file {} does not exist'.format(file_path))

        with open(file_path, 'rb') as f:
            hyperparameters = pickle.load(f)
            self.batch_size = hyperparameters.pop('batch_size', self.batch_size)
            self.dropout = hyperparameters.pop('dropout', self.dropout)
            self.embedding_dim = hyperparameters.pop('embedding_dim', self.embedding_dim)
            self.epoch = hyperparameters.pop('epoch', self.epoch)
            self.grad_clip_value = hyperparameters.pop('grad_clip_value', self.grad_clip_value)
            self.hidden_dim = hyperparameters.pop('hidden_dim', self.hidden_dim)
            self.learning_rate = hyperparameters.pop('learning_rate', self.learning_rate)
            self.min_count = hyperparameters.pop('min_count', self.min_count)
            self.num_rnn_layers = hyperparameters.pop('num_rnn_layers', self.num_rnn_layers)
            self.seed = hyperparameters.pop('seed', self.seed)

        return self

    def save_to_file(self, file_path=None):
        if file_path is None or type(file_path) != str:
            raise ValueError('argument `file_path` should be a string')
        else:
            with open(file_path, 'wb') as f:
                hyperparameters = {
                    'batch_size': self.batch_size,
                    'dropout': self.dropout,
                    'embedding_dim': self.embedding_dim,
                    'epoch': self.epoch,
                    'grad_clip_value': self.grad_clip_value,
                    'hidden_dim': self.hidden_dim,
                    'learning_rate': self.learning_rate,
                    'min_count': self.min_count,
                    'num_rnn_layers': self.num_rnn_layers,
                    'seed': self.seed,
                }

                pickle.dump(hyperparameters, f)
        return self