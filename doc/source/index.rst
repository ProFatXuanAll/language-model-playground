Welcome to Language Model Playground's documentation!
=====================================================

**Language model playground** is a tutorial about "How to implement :term:`neural network` based
:term:`language models`".  We use Pytorch_ to implement :term:`language models`.

.. _PyTorch: https://pytorch.org/

We have implemented several :term:`language models` including:

- RNN only models.
  (e.g. :py:mod:`lmp.model.ElmanNet`)
- RNN models with residual connections.
  (e.g. :py:mod:`lmp.model.ResElmanNet`)
- RNN models with self attention mechanism.
  (e.g. :py:mod:`lmp.model.SAttnElmanNet`)
- Transformer models.
- And more to come!

You can easily create these models by ``import lmp.model`` and choose the model you want.  Or you can create models
using training script :py:mod:`lmp.script.train_model`.

.. code-block:: python

   import lmp.model

   model = lmp.model.ElmanNet(...)      # parameters go in here.

.. seealso::

   :py:mod:`lmp.model`
      All available language models are put under :py:mod:`lmp.model`.

.. todo::

   Add Transformer models.

We have written serveral **scripts** to demostrate training pipline of :term:`language models` and furthur usage on
:term:`language models`:

- Use :py:mod:`lmp.script.sample_dset` to take a look at dataset we provided.
  (e.g. :py:class:`lmp.dset.WikiText2Dset`)
- Use :py:mod:`lmp.script.train_tknzr` to train :term:`tokenizers`.
  (e.g. :py:class:`lmp.tknzr.WsTknzr`)
- Use :py:mod:`lmp.script.tknz_txt` to :term:`tokenize` text with pre-trained :term:`tokenizers`.
- Use :py:mod:`lmp.script.train_model` to train :term:`language models`.
- Use :py:mod:`lmp.script.eval_txt_ppl` to calculate :term:`perplexity` on given sample with pre-trained
  :term:`language model` checkpoint.
- Use :py:mod:`lmp.script.eval_dset_ppl` to calculate :term:`perplexity` on dataset with range of
  pre-trained :term:`language model` :term:`checkpoints`.
- Use :py:mod:`lmp.script.gen_txt` to generate text with pre-trained :term:`language model` :term:`checkpoint`.
- And more to come!

Get started with :doc:`Quick start <quickstart>` or jump directly to contents you are interesting in!

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
