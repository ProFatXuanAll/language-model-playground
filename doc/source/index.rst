.. Language Model Playground documentation master file, created by
    sphinx-quickstart on Fri Nov 20 15:35:55 2020.
    You can adapt this file completely to your liking, but it should at least
    contain the root `toctree` directive.

Welcome to Language Model Playground's documentation!
=====================================================

Language model playground is a tutorial of "How to implement neural network
based language models".
We use Pytorch_ to implement language models.
We have implemented several language models including:

- RNN only models.
- RNN models with attention mechanism.
- Transformer models.

.. _PyTorch: https://pytorch.org/

We have written serveral scripts to demostrate training pipline of language
models and furthur usage on language models:

- Use :py:mod:`lmp.script.sample_from_dataset` to take a look at dataset we
  provided.
- Use :py:mod:`lmp.script.train_tokenizer` to train tokenizers.
- Use :py:mod:`lmp.script.tokenize` to tokenize text with pre-trained
  tokenizers.
- Use :py:mod:`lmp.script.train_model` to train language models.
- Use ``TODO`` to validate pre-trained language models with perplexity.
- Use ``TODO`` to generate text with pre-trained language models.
- And more to come!

Get started with :doc:`Quick start <quickstart>` or jump directly to contents
you are intresting in!

.. toctree::
    :maxdepth: 2
    :caption: Table of Contents:

    quickstart
    lmp/index
    contribute
    how_to_doc
    how_to_test
    glossary

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
