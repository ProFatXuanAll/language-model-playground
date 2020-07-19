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
        checkpoint_size:
            Checkpoint interval based on number of mini-batch.
            Must be bigger than or equal to `1`.
        dropout:
            Dropout rate.
            Range [0 , 1]
        embedding_dim:
            Embedding dimension.
            Must be bigger than or equal to `1`.
        epoch:
            Number of training epochs.
            epoch must be bigger than or equal to '1'
        hidden_dim:
            Hidden dimension.
            Must be bigger than or equal to `1`.
        is_uncased:
            Convert all upper case to lower case.
            Must be True or False.
        learning_rate:
            Optimizer's parameter `lr`.
            Must be bigger than `0`.
        max_norm:
            Max norm of gradient.
            Used when cliping gradient norm.
            Must be bigger than `0`.
        min_count:
            Minimum of token'sfrequence.
            Used to filter words that is smaller than min_count.
        model_type:
            Decide to use which model, LSTM or GRU.
        num_rnn_layers:
            Number of rnn layers.
            Must be bigger than or equal to `1`.
        num_linear_layers
            Number of Linear layers.
            Must be bigger than or equal to `1`.
        seed:
            Control random seed.
            Must be bigger than `0`.
        tokenizer_type:
            Decide to use which tokenizer, list or dict
            Tokenizer's token_to_id is implemented in different structure(list or dict).

    """

    def __init__(
            self,
            batch_size: int = 1,
            checkpoint_size: int = 500,
            dropout: float = 0,
            embedding_dim: int = 1,
            epoch: int = 1,
            hidden_dim: int = 1,
            is_uncased: bool = False,
            learning_rate: float = 5e-5,
            max_norm: float = 1,
            min_count: int = 0,
            model_type: str = 'lstm',
            num_rnn_layers: int = 1,
            num_linear_layers: int = 1,
            seed: int = 1,
            tokenizer_type: str = 'list'
    ):

        self.batch_size = batch_size
        self.checkpoint_size = checkpoint_size
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.epoch = epoch
        self.hidden_dim = hidden_dim
        self.is_uncased = is_uncased
        self.learning_rate = learning_rate
        self.max_norm = max_norm
        self.min_count = min_count
        self.model_type = model_type
        self.num_rnn_layers = num_rnn_layers
        self.num_linear_layers = num_linear_layers
        self.seed = seed
        self.tokenizer_type = tokenizer_type

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
            return cls(**pickle.load(f))

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
                    'checkpoint_size': self.checkpoint_size,
                    'dropout': self.dropout,
                    'embedding_dim': self.embedding_dim,
                    'epoch': self.epoch,
                    'hidden_dim': self.hidden_dim,
                    'is_uncased': self.is_uncased,
                    'learning_rate': self.learning_rate,
                    'max_norm': self.max_norm,
                    'min_count': self.min_count,
                    'model_type': self.model_type,
                    'num_rnn_layers': self.num_rnn_layers,
                    'num_linear_layers': self.num_linear_layers,
                    'seed': self.seed,
                    'tokenizer_type': self.tokenizer_type
                }

                pickle.dump(hyperparameters, f)
        return self
