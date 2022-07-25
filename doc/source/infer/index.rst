Inference methods
=================

Overview
--------
An inference method can let a :term:`language model` generate continual text of given text.
One must provide a language model, its paired :term:`tokenizer` and a text to an inference method.
The generated results will returned as text.

.. seealso::

  :doc:`lmp.model </model/index>`
    All available language models.
  :doc:`lmp.script.gen_txt </script/gen_txt>`
    Continual text generation script.
  :doc:`lmp.tknzr </tknzr/index>`
    All available tokenizers.

Import inference method module
------------------------------
All inference method classes are collectively gathered under the module :py:mod:`lmp.infer`.
One can import inference method module as usual Python_ module:

.. code-block:: python

  import lmp.infer

Create inference method instance
--------------------------------
After importing :py:mod:`lmp.infer`, one can create inference method instance through the class attributes of :py:mod:`lmp.infer`.
For example, one can create top-1 inference method :py:class:`lmp.infer.Top1Infer` and top-K inference method :py:class:`lmp.infer.TopKInfer` as follow:

.. code-block:: python

  import lmp.infer

  # Create top-1 inference method instance.
  top_1_infer = lmp.infer.Top1Infer(max_seq_len=512)

  # Create top-K inference method instance.
  top_k_infer = lmp.infer.TopKInfer(k=5, max_seq_len=512)

The ``max_seq_len`` parameters is provided to avoid non-stopping generation.
The maximum value of ``max_seq_len`` is ``1024``.
Different inference methods have different parameters.
For example, top-K inference method has a parameter ``k`` which represent the first ``k`` possible tokens to sample.

Generate continual text
-----------------------
To generate continual text on a given text, one must provide a :term:`language model`, its paired :term:`tokenizer` and a text to an inference method.
Only :term:`pre-trained` language model can generate meanful text.
The following example demonstrate the usage of generation without pre-training a language model.

.. code-block:: python

  import lmp.infer
  import lmp.model
  import lmp.tknzr

  inference = lmp.infer.Top1Infer(max_seq_len=512)
  tokenizer = lmp.tknzr.CharTknzr(
    is_uncased=True,
    max_vocab=-1,
    min_count=0,
  )
  model = lmp.model.ElmanNet(
    d_emb=10,
    d_hid=10,
    n_lyr=1,
    p_emb=0.1,
    p_hid=0.1,
    tknzr=tokenizer,
  )

  # Generate continual text.
  generated_txt = inference.gen(model=model, tknzr=tokenizer, txt='abc')

All available inference methods
-------------------------------

.. toctree::
  :glob:
  :maxdepth: 1

  *

.. _Python: https://www.python.org/
