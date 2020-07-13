r"""Configuration for text-generation experiment.
Usage:
    config = lmp.config.BaseConfig(...params)
    config.save_to_file(path)
    config = config.load_from_file(path)
"""

# built-in modules
import os
import pickle

class BaseConfig:
    r"""Configuration for text-generation model.

    Attributes:
        batch_size:
            Training batch size.
            default is 1
        dropout:
            Dropout rate.
            Range [0 , 1]
        embedding_dim:
            Embedding dimension.
            Must be bigger than or equal to `1`.
        epoch:
            Number of training epochs.
            epoch must be bigger than or equal to '1'
        max_norm:
            Max norm of gradient.
            Used when cliping gradient norm.
            Must be bigger than `0`.
        hidden_dim:
            Hidden dimension.
            Must be bigger than or equal to `1`.
        learning_rate:
            Optimizer's parameter `lr`.
            Must be bigger than `0`.
        min_count:
            Minimum of token'sfrequence.
            Used to filter words that is smaller than min_count.
        num_rnn_layers:
            Number of rnn layers.
            Must be bigger than or equal to `1`.
        num_linear_layers
            Number of Linear layers.
            Must be bigger than or equal to `1`.
        seed:
            Control random seed.
            Must be bigger than `0`.
        is_uncased:
            Convert all upper case to lower case.
            Must be True or False.
        checkpoint:
            Checkpoint interval based on number of mini-batch.
            Must be bigger than or equal to `1`.

    """
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
            seed: int = 1,
            is_uncased: bool = False,
            checkpoint: int = 500
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
        self.is_uncased = is_uncased
        self.checkpoint = checkpoint

    @classmethod
    def load_from_file(cls, file_path: str = None):
        r"""Load configuration from pickle  file.
        Args:
            file_path: Location of pickle file.
        """
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
            self.seed = hyperparameters.pop('is_uncased', self.is_uncased)
            self.checkpoint = hyperparameters.pop('checkpoint', self.checkpoint)

        return self

    def save_to_file(self, file_path: str = None):
        r"""Save configuration into pickle file.
        Args:
            file_path: Location of saving file.
        """
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
                    'is_uncased': self.is_uncased,
                    'checkpoint': self.checkpoint
                }

                pickle.dump(hyperparameters, f)
        return self