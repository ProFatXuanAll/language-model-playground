Language Models
===============

Overview
--------
In this project, a :term:`language model` is a deep learning :term:`model` which can predict next possible token
conditioned on given tokens.
Each language model has a :term:`loss function` which can be :term:`optimized` by the language model :term:`training`
script :doc:`lmp.script.train_model </script/train_model>`.
The goal of optimizing a language model objective function is to make a language model have low :term:`perplexity`,
which serve as an indication of performing well on next token prediction.
A language model must be paired with a :term:`tokenizer`, and once paired with a tokenizer it cannot change to pair
with another tokenizer.
A tokenizer will share its vocabulary with a language model.
When construct a language model, one must first construct a tokenizer and then pass that tokenizer to model constructor.

.. seealso::

  :doc:`lmp.tknzr </tknzr/index>`
    All available tokenizers.
  :doc:`lmp.script.eval_dset_ppl </script/eval_dset_ppl>`
    Dataset perplexity evaluation script.
  :doc:`lmp.script.eval_txt_ppl </script/eval_txt_ppl>`
    Text perplexity evaluation script.
  :doc:`lmp.script.gen_txt </script/gen_txt>`
    Continual text generation script.
  :doc:`lmp.script.train_model </script/train_model>`
    Language model training script.

Import language model module
----------------------------
All :term:`language model` classes are collectively gathered under the module :py:mod:`lmp.model`.
One can import language model module as usual Python_ module:

.. code-block:: python

  import lmp.model

Create language model instance
------------------------------
After importing :py:mod:`lmp.model`, one can create :term:`language model` instance through the class attributes of
:py:mod:`lmp.model`.
For example, one can create Elman-Net language model :py:class:`lmp.model.ElmanNet` as follow:

.. code-block:: python

  import lmp.model
  import lmp.tknzr

  # Create tokenizer instance.
  tokenizer = lmp.tknzr.CharTknzr(
    is_uncased=False,
    max_vocab=-1,
    min_count=0,
  )
  # Create language model instance.
  model = lmp.model.ElmanNet(
    d_emb=100,
    d_hid=300,
    p_emb=0.1,
    p_hid=0.1,
    tknzr=tokenizer,
  )

Each language model is an instance of :py:class:`torch.nn.Module`.
Each language model must be paired with a :term:`tokenizer`.
In the example above we show that an Elman-Net language model can be paired with a character tokenizer.

Initialize language model parameters
------------------------------------
Pytorch_ provides built-in utilities to initialize :term:`model parameters`.
All initialization utilities are collectively gathered under the module :py:mod:`torch.nn.init`.

.. code-block:: python

  import lmp.model
  import lmp.tknzr
  import torch

  model = lmp.model.ElmanNet(
    d_emb=100,
    d_hid=300,
    p_emb=0.1,
    p_hid=0.1,
    tknzr=lmp.tknzr.CharTknzr(
      is_uncased=False,
      max_vocab=-1,
      min_count=0,
    ),
  )

  # Initialize model parameters.
  torch.nn.init.zeros_(model.h_0)

If you cannot decide how to initialize a :term:`language model`, we have provided an utility :py:meth:`params_init` for
each language model to help you initialize model parameters.

.. code-block:: python

  import lmp.model
  import lmp.tknzr

  model = lmp.model.ElmanNet(
    d_emb=100,
    d_hid=300,
    p_emb=0.1,
    p_hid=0.1,
    tknzr=lmp.tknzr.CharTknzr(
      is_uncased=False,
      max_vocab=-1,
      min_count=0,
    ),
  )

  # Initialize model parameters.
  model.params_init()

Calculate loss function
-----------------------
One can calculate :term:`mini-batch` :term:`loss` of a :term:`language model` using :py:meth:`loss` function.
For example,

.. code-block:: python

  import lmp.model
  import lmp.tknzr
  import torch

  tokenizer = lmp.tknzr.CharTknzr(
    is_uncased=False,
    max_vocab=-1,
    min_count=0,
  )
  model = lmp.model.ElmanNet(
    d_emb=100,
    d_hid=300,
    p_emb=0.1,
    p_hid=0.1,
    tknzr=tokenizer,
  )

  # Encode mini-batch.
  batch_txt = ['hello world', 'how are you']
  tokenizer.build_vocab(batch_txt=batch_txt)
  batch_tkids = tokenizer.batch_enc(batch_txt=batch_txt, max_seq_len=15)
  batch_tkids = torch.LongTensor(batch_tkids)

  # Calculate mini-batch loss.
  loss, batch_cur_states = model.loss(
    batch_cur_tkids=batch_tkids[:, :-1],
    batch_next_tkids=batch_tkids[:, 1:],
    batch_prev_states=None,
  )

The method :py:meth:`loss` takes three input and returns a tuple.
The ``batch_cur_tkids`` is the input :term:`token id` list and the ``batch_next_tkids`` is the prediction target.
Both ``batch_cur_tkids`` and ``batch_next_tkids`` are long tensor and have the same shape :math:`(B, S)` where
:math:`B` is the :term:`batch size` and :math:`S` is input sequence length.
We set ``batch_prev_states=None`` to use :term:`initial hidden states`.
The first item in the returned tuple is a :py:class:`torch.Tensor` which represents the mini-batch next token
prediction loss.
One can call the PyTorch_ built-in ``backward`` method to perform :term:`back-propagation`.
The second item in the returned tuple is a list of :py:class:`torch.Tensor` which represents the current
:term:`hidden states` of a language model.
The current hidden state can be used as the initial hidden states of next input.
This is usually used when one can only process certain sequence length at a time.
Each chucking sequence is called a :term:`context window`.
For example, we can split the input in the above example into half as follow:

.. code-block:: python

  import lmp.model
  import lmp.tknzr
  import torch

  tokenizer = lmp.tknzr.CharTknzr(
    is_uncased=False,
    max_vocab=-1,
    min_count=0,
  )
  model = lmp.model.ElmanNet(
    d_emb=100,
    d_hid=300,
    p_emb=0.1,
    p_hid=0.1,
    tknzr=tokenizer,
  )

  # Encode mini-batch.
  batch_txt = ['hello world', 'how are you']
  tokenizer.build_vocab(batch_txt=batch_txt)
  batch_tkids = tokenizer.batch_enc(batch_txt=batch_txt, max_seq_len=15)
  batch_tkids = torch.LongTensor(batch_tkids)

  # Calculate mini-batch first-half loss.
  loss_1, batch_cur_states = model.loss(
    batch_cur_tkids=batch_tkids[:, 0:8],
    batch_next_tkids=batch_tkids[:, 1:9],
    batch_prev_states=None,
  )

  # Perform back-propagation.
  loss_1.backward()

  # Calculate mini-batch second-half loss.
  loss_2, _ = model.loss(
    batch_cur_tkids=batch_tkids[:, 8:-1],
    batch_next_tkids=batch_tkids[:, 9:],
    batch_prev_states=batch_cur_states,
  )

  # Perform back-propagation.
  loss_2.backward()

Predict next token
------------------
Next token prediction can be done by the ``pred`` method.
The input of ``pred`` is almost the same as ``loss``, except that we do not input the prediction target.
This is because when performing evaluation one do not and cannot know the prediction target.
One set ``batch_prev_states=None`` to use :term:`initial hidden states` just like using ``loss``.
The returned tuple have two items.
The first item in the returned tuple is a :py:class:`torch.Tensor` which represent the next :term:`token id`
probability distribution.
The probability distribution tensor has shape :math:`(B, S, V)` where :math:`B` is :term:`batch size`, :math:`S` is
input sequence length and :math:`V` is the :term:`vocabulary` size of the :term:`language model` pairing
:term:`tokenizer`.
The second item in the returned tuple is a list of :py:class:`torch.Tensor` which represents the current
:term:`hidden states` of a language model.
Just like the ``loss`` method, the current hidden states can be used as the :term:`initial hidden states` of next input.

.. code-block:: python

  import lmp.model
  import lmp.tknzr
  import torch

  tokenizer = lmp.tknzr.CharTknzr(
    is_uncased=False,
    max_vocab=-1,
    min_count=0,
  )
  model = lmp.model.ElmanNet(
    d_emb=100,
    d_hid=300,
    p_emb=0.1,
    p_hid=0.1,
    tknzr=tokenizer,
  )

  # Encode mini-batch.
  batch_txt = ['hello world', 'how are you']
  tokenizer.build_vocab(batch_txt=batch_txt)
  batch_tkids = tokenizer.batch_enc(batch_txt=batch_txt, max_seq_len=15)
  batch_tkids = torch.LongTensor(batch_tkids)

  # Calculate next token prediction.
  pred, batch_cur_states = model.pred(
    batch_cur_tkids=batch_tkids,
    batch_prev_states=None,
  )

All available language models
-----------------------------
.. toctree::
  :glob:
  :maxdepth: 1

  *

.. _Python: https://www.python.org/
.. _PyTorch: https://pytorch.org/
