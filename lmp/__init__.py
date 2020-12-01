r"""Language model playground.

This module provide utilities for training language model, evaluate perplexity
for language model and generate sequences using language model. Submodule
`lmp.config` setup configuration for language model experiment. Submodule
`lmp.dataset` setup dataset in language model format. Submodule `lmp.model`
define language model architecture. Submodule `lmp.path` define path variables
shared by all files in `lmp` module. Submodule `lmp.tknzr` define tokenizer
for language model dataset preprocessing. Submodule `lmp.util` provide utilites
for language model training, evalutate and inference.

Usage:
    import lmp
"""


import lmp.model
import lmp.path
import lmp.tknzr
import lmp.util
