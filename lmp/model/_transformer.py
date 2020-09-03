r"""Seq2seq model with encoder and decoder.

Usage:
    import lmp

    model = lmp.model.transformer(...)
    logits = model(...)

"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# 3rd-party modules

import torch
import torch.nn

