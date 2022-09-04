Language Models
===============

Overview
--------
In this project, a :term:`language model` is a deep learning :term:`model` which can predict next possible token conditioned on given tokens.
Each language model has a :term:`loss function` which can be :term:`optimized` by the language model :term:`training` script :doc:`lmp.script.train_model </script/train_model>`.
The optimization goal of a language model is to have low :term:`perplexity`, which serve as an indication of performing well on next token prediction.
A language model is paired with one and only one :term:`tokenizer`.
A language model always predict tokens contained in the paired tokenizer's :term:`vocabulary`.
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

Create language model instances
-------------------------------
After importing :py:mod:`lmp.model`, one can create :term:`language model` instance through the class attributes of :py:mod:`lmp.model`.
For example, one can create Elman-Net language model :py:class:`~lmp.model.ElmanNet` as follow:

.. code-block:: python

  import lmp.model
  import lmp.tknzr

  # Create tokenizer instance.
  tokenizer = lmp.tknzr.CharTknzr()

  # Create language model instance.
  model = lmp.model.ElmanNet(tknzr=tokenizer)

Each language model is an instance of :py:class:`torch.nn.Module`.
Each language model is paired with one and only one :term:`tokenizer`.
In the example above we see that an Elman-Net language model can be paired with a character tokenizer.

Initialize language model parameters
------------------------------------
Pytorch_ provides built-in utilities to initialize :term:`model parameters`.
All initialization utilities are collectively gathered under the module :py:mod:`torch.nn.init`.

.. code-block:: python

  import lmp.model
  import lmp.tknzr
  import torch

  # Create language model instance.
  model = lmp.model.ElmanNet(tknzr=lmp.tknzr.CharTknzr())

  # Initialize model parameters.
  torch.nn.init.zeros_(model.fc_e2h.bias)

If you cannot decide how to initialize a :term:`language model`, we have provided an utility :py:meth:`~lmp.model.BaseModel.params_init` for each language model to help you initialize model parameters.

.. code-block:: python

  import lmp.model
  import lmp.tknzr

  # Create language model instance.
  model = lmp.model.ElmanNet(tknzr=lmp.tknzr.CharTknzr())

  # Initialize model parameters.
  model.params_init()

Calculate prediction loss
-------------------------
One can calculate :term:`mini-batch` :term:`loss` of a :term:`language model` using :py:meth:`~lmp.model.BaseModel.cal_loss` function.
For example,

.. code-block:: python

  import lmp.model
  import lmp.tknzr
  import torch

  # Create tokenizer instance.
  tokenizer = lmp.tknzr.CharTknzr()

  # Build tokenizer vocabulary.
  batch_txt = ['hello world', 'how are you']
  tokenizer.build_vocab(batch_txt=batch_txt)

  # Encode mini-batch.
  batch_tkids = []
  for txt in batch_txt:
    tkids = tokenizer.enc(txt=txt)
    tkids = tokenizer.pad_to_max(max_seq_len=20, tkids=tkids)
    batch_tkids.append(tkids)

  # Convert mini-batch to tensor.
  batch_tkids = torch.LongTensor(batch_tkids)

  # Create language model instance.
  model = lmp.model.ElmanNet(tknzr=tokenizer)

  # Calculate mini-batch loss.
  loss, batch_cur_states = model.cal_loss(
    batch_cur_tkids=batch_tkids[:, :-1],
    batch_next_tkids=batch_tkids[:, 1:],
    batch_prev_states=None,
  )

The method :py:meth:`~lmp.model.BaseModel.cal_loss` takes three input and returns a tuple.
The ``batch_cur_tkids`` is the input :term:`token id` list and the ``batch_next_tkids`` is the prediction target.
Both ``batch_cur_tkids`` and ``batch_next_tkids`` are long tensor and have the same shape :math:`(B, S)` where :math:`B` is the :term:`batch size` and :math:`S` is input sequence length.
We set ``batch_prev_states=None`` to use :term:`initial hidden states`.
The first item in the returned tuple is a :py:class:`torch.Tensor` which represents the mini-batch next token prediction loss.
One can call the PyTorch_ built-in :py:meth:`torch.Tensor.backward` method to perform :term:`back-propagation`.
The second item in the returned tuple represents the current :term:`hidden states` of a language model.
The exact structure of current hidden states depends on which language model is used.
The current hidden states can be used as the initial hidden states of next input.
This is done by pass current hidden states as ``batch_prev_states``.
This is needed since one can only process certain sequence length at a time.

Predict next token
------------------
Next token prediction can be done by the :py:meth:`~lmp.model.BaseModel.pred` method.
The input of :py:meth:`~lmp.model.BaseModel.pred` is almost the same as :py:meth:`~lmp.model.BaseModel.cal_loss`, except that we do not input the prediction target.
This is because when performing evaluation one do not and cannot know the prediction target.
One set ``batch_prev_states=None`` to use :term:`initial hidden states` just as in :py:meth:`~lmp.model.BaseModel.cal_loss`.
The returned tuple have two items.
The first item in the returned tuple is a :py:class:`torch.Tensor` which represent the next :term:`token id` probability distribution.
The probability distribution tensor has shape :math:`(B, S, V)` where :math:`B` is :term:`batch size`, :math:`S` is input sequence length and :math:`V` is the :term:`vocabulary` size of the :term:`language model` pairing :term:`tokenizer`.
The second item in the returned tuple represents the current :term:`hidden states` of a language model.
One should compare this with :py:meth:`~lmp.model.BaseModel.cal_loss`.

.. code-block:: python

  import lmp.model
  import lmp.tknzr
  import torch

  # Create tokenizer instance.
  tokenizer = lmp.tknzr.CharTknzr()

  # Build tokenizer vocabulary.
  batch_txt = ['hello world', 'how are you']
  tokenizer.build_vocab(batch_txt=batch_txt)

  # Encode mini-batch.
  batch_tkids = []
  for txt in batch_txt:
    tkids = tokenizer.enc(txt=txt)
    tkids = tokenizer.pad_to_max(max_seq_len=20, tkids=tkids)
    batch_tkids.append(tkids)

  # Convert mini-batch to tensor.
  batch_tkids = torch.LongTensor(batch_tkids)

  # Create language model instance.
  model = lmp.model.ElmanNet(tknzr=tokenizer)

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

.. footbibliography::

.. _Python: https://www.python.org/
.. _PyTorch: https://pytorch.org/
