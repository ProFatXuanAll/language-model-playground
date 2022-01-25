Welcome to Language Model Playground's documentation!
=====================================================

**Language model playground** is a tutorial about "How to implement :term:`neural network` based
:term:`language models`".  We use Pytorch_ to implement language models.

We have implemented several :term:`language models` including:

- Elman Net. (See :py:mod:`lmp.model.ElmanNet`.)
- LSTM and its variations.  (See :py:mod:`lmp.model.LSTM1997`, :py:mod:`lmp.model.LSTM2000`.)
- And more to come!

You can easily create these models instance using module :py:mod:`lmp.model`.  You can also train these models directly
using CLI script :py:mod:`lmp.script.train_model`.

.. code-block:: python

   import lmp.model

   model = lmp.model.ElmanNet(...)  # parameters go in here.
   model = lmp.model.LSTM1997(...)  # parameters go in here.

.. seealso::

   :py:mod:`lmp.model`
      All available language models are put under :py:mod:`lmp.model`.

.. todo::

   Add Transformer models.

We have written serveral **scripts** to demonstrate typical training pipline of :term:`language models` and demonstrate
furthur usage on language models:

- Use :py:mod:`lmp.script.sample_dset` to take a look at available datasets.
- Use :py:mod:`lmp.script.train_tknzr` to train :term:`tokenizers`.
- Use :py:mod:`lmp.script.tknz_txt` to :term:`tokenize` text with pre-trained tokenizers.
- Use :py:mod:`lmp.script.train_model` to train language models.
- Use :py:mod:`lmp.script.eval_txt_ppl` to calculate :term:`perplexity` on the given sample with pre-trained
  language model :term:`checkpoint`.
- Use :py:mod:`lmp.script.eval_dset_ppl` to calculate perplexity on dataset with range of pre-trained language model
  checkpoints.
- Use :py:mod:`lmp.script.gen_txt` to generate continual text with pre-trained language model checkpoint.

See :doc:`quick start <quickstart>` for typical language model training pipline, or jump directly to contents you are
interesting in!

.. toctree::
   :maxdepth: 2
   :caption: Table of Contents:

   quickstart
   lmp/index
   experiment/index
   contribute
   how_to_doc
   how_to_test
   glossary

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _PyTorch: https://pytorch.org/
