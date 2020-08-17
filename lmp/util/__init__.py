r"""Helper functions collection.

All utilities files in this module must re-import in this very file. This help
to avoid unnecessary import structure (for example, we prefer using
`lmp.util._load_model` over `lmp.util._model.load_model`).

All submodules which provide loading utilites should provide two interface, one
for directly passing parameter and one for using configuration object (for
example, `lmp.util._load_model` and `lmp.util._load_model_by_config`).

Usage:
    import lmp.util

    dataset = lmp.util.load_dataset_by_config(...)
    tokenizer = lmp.util.train_tokenizer_by_config(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# self-made modules

from lmp.util._config import load_config
from lmp.util._dataset import load_dataset
from lmp.util._dataset import load_dataset_by_config
from lmp.util._perplexity_eval import perplexity_eval
from lmp.util._perplexity_eval import batch_perplexity_eval
from lmp.util._generate_sequence import generate_sequence
from lmp.util._generate_sequence import generate_sequence_by_config
from lmp.util._model import load_model
from lmp.util._model import load_model_by_config
from lmp.util._optimizer import load_optimizer
from lmp.util._optimizer import load_optimizer_by_config
from lmp.util._seed import set_seed
from lmp.util._seed import set_seed_by_config
from lmp.util._tokenizer import load_tokenizer
from lmp.util._tokenizer import load_tokenizer_by_config
from lmp.util._train_model import train_model
from lmp.util._train_model import train_model_by_config
from lmp.util._train_tokenizer import train_tokenizer
from lmp.util._train_tokenizer import train_tokenizer_by_config
