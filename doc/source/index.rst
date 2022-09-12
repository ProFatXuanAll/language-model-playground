Welcome to Language Model Playground's documentation!
=====================================================

**Language model playground** is a tutorial about "How to implement :term:`neural network` based :term:`language models`".
We use Pytorch_ to implement language models.

We have implemented several :term:`language models` including:

- Elman Net.
  (See :py:class:`~lmp.model.ElmanNet`.)
- LSTM and its variations.
  (See :py:class:`~lmp.model.LSTM1997`, :py:class:`~lmp.model.LSTM2000` and :py:class:`~lmp.model.LSTM2002`.)
- Transformer encoder.
  (See :py:class:`~lmp.model.TransEnc`)
- And more to come!

You can easily create these models instance using module :py:mod:`lmp.model`.
You can also train these models directly using CLI script :doc:`lmp.script.train_model </script/train_model>`.

.. code-block:: python

  import lmp.model

  model = lmp.model.ElmanNet(...)  # parameters go in here.
  model = lmp.model.LSTM1997(...)  # parameters go in here.
  model = lmp.model.TransEnc(...)  # parameters go in here.

.. seealso::

  :doc:`lmp.model </model/index>`
    All available language models.

We have written serveral **scripts** to demonstrate typical training pipline of :term:`language models` and demonstrate furthur usage on language models:

- Use :doc:`lmp.script.sample_dset </script/sample_dset>` to take a look at available datasets.
- Use :doc:`lmp.script.train_tknzr </script/train_tknzr>` to train :term:`tokenizers`.
- Use :doc:`lmp.script.tknz_txt </script/tknz_txt>` to :term:`tokenize` text with pre-trained tokenizers.
- Use :doc:`lmp.script.train_model </script/train_model>` to train language models.
- Use :doc:`lmp.script.eval_txt_ppl </script/eval_txt_ppl>` to calculate :term:`perplexity` on the given sample with pre-trained language model :term:`checkpoint`.
- Use :doc:`lmp.script.eval_dset_ppl </script/eval_dset_ppl>` to calculate perplexity on dataset over a range of pre-trained language model checkpoints.
- Use :doc:`lmp.script.gen_txt </script/gen_txt>` to generate continual text with pre-trained language model checkpoint.

See :doc:`quick start <quickstart>` for typical language model training pipline, or jump directly to contents you are interesting in!

.. toctree::
  :maxdepth: 2
  :caption: Table of Contents:

  quickstart
  script/index
  dset/index
  tknzr/index
  model/index
  infer/index
  experiment/index
  dev/index
  glossary

Indices and tables
==================

- :ref:`genindex`
- :ref:`modindex`
- :ref:`search`

.. footbibliography::

.. _PyTorch: https://pytorch.org/
