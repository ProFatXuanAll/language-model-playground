# built-in modules
import os
import pickle

class BaseConfig:
    def __init__(
            self, 
            batch_size: int = 1,
            dropout: float = 0,
            embedding_dim: int = 1,
            epoch: int = 1,
            max_norm: float = 1,
            hidden_dim: int = 1,
            learning_rate: float = 10e-4,
            min_count: int = 0,
            num_rnn_layers: int = 1,
            num_linear_layers: int = 1,
            seed: int = 1
    ):

        self.batch_size = batch_size
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.epoch = epoch
        self.max_norm = max_norm
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.num_rnn_layers = num_rnn_layers
        self.num_linear_layers = num_linear_layers
        self.seed = seed

    @classmethod
    def load_from_file(cls, file_path: str = None):
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
            self.max_norm = hyperparameters.pop('max_norm', self.max_norm)
            self.hidden_dim = hyperparameters.pop('hidden_dim', self.hidden_dim)
            self.learning_rate = hyperparameters.pop('learning_rate', self.learning_rate)
            self.min_count = hyperparameters.pop('min_count', self.min_count)
            self.num_rnn_layers = hyperparameters.pop('num_rnn_layers', self.num_rnn_layers)
            self.num_linear_layers = hyperparameters.pop('num_linear_layers', self.num_linear_layers)
            self.seed = hyperparameters.pop('seed', self.seed)

        return self

    def save_to_file(self, file_path: str = None):
        if file_path is None or type(file_path) != str:
            raise ValueError('argument `file_path` should be a string')
        else:
            with open(file_path, 'wb') as f:
                hyperparameters = {
                    'batch_size': self.batch_size,
                    'dropout': self.dropout,
                    'embedding_dim': self.embedding_dim,
                    'epoch': self.epoch,
                    'max_norm': self.max_norm,
                    'hidden_dim': self.hidden_dim,
                    'learning_rate': self.learning_rate,
                    'min_count': self.min_count,
                    'num_rnn_layers': self.num_rnn_layers,
                    'num_linear_layers': self.num_linear_layers,
                    'seed': self.seed,
                }

                pickle.dump(hyperparameters, f)
        return self