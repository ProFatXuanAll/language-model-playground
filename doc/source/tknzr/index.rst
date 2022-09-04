Tokenizers
==========

Overview
--------
In this project, a :term:`tokenizer` is a collection of text preprocessing tools including text :term:`tokenization`, :term:`text normalization` and :term:`language model` :term:`training` formation.
To obtain a tokenizer, one must first create an empty tokenizer, then pick a :term:`dataset` to train the chosen tokenizer.

.. seealso::

  :doc:`lmp.dset </dset/index>`
    All available datasets.
  :doc:`lmp.script.tknz_txt </script/tknz_txt>`
    Text tokenization script.
  :doc:`lmp.script.train_tknzr </script/train_tknzr>`
    Tokenizer training script.

Import tokenizer module
-----------------------
All :term:`tokenizer` classes are collectively gathered under the module :py:mod:`lmp.tknzr`.
One can import tokenizer module as usual Python_ module:

.. code-block:: python

  import lmp.tknzr

Create tokenizer instances
--------------------------
After importing :py:mod:`lmp.tknzr`, one can create :term:`tokenizer` instance through the class attributes of :py:mod:`lmp.tknzr`.
For example, one can create character tokenizer :py:class:`lmp.tknzr.CharTknzr` and whitespace tokenizer :py:class:`lmp.tknzr.WsTknzr` as follow:

.. code-block:: python

  import lmp.tknzr

  # Create character tokenizer instance.
  character_tokenizer = lmp.tknzr.CharTknzr()

  # Create whitespace tokenizer instance.
  whitespace_tokenizer = lmp.tknzr.WsTknzr()

Tokenize text
-------------
A :term:`tokenizer` instance can, of course, tokenize text.
For example, a character tokenizer can tokenize text into a list of characters, and a whitespace tokenizer can tokenize text into a list of whitespace-separated tokens (words in the context of plaint English):

.. code-block:: python

  import lmp.tknzr

  # Create tokenizer instance.
  character_tokenizer = lmp.tknzr.CharTknzr()
  whitespace_tokenizer = lmp.tknzr.WsTknzr()

  # Tokenize text into token list.
  assert character_tokenizer.tknz(txt='abc') == ['a', 'b', 'c']
  assert whitespace_tokenizer.tknz(txt='a b c') == ['a', 'b', 'c']

Each :term:`special token` will be treated as an unit no matter which tokenizer is used.

.. code-block:: python

  import lmp.tknzr
  from lmp.vars import BOS_TK, EOS_TK

  # Create tokenizer instance.
  tokenizer = lmp.tknzr.CharTknzr()

  # Tokenize text into token list.
  assert tokenizer.tknz(txt=f'{BOS_TK}abc{EOS_TK}') == [BOS_TK, 'a', 'b', 'c', EOS_TK]

Detokenize token list
---------------------
:term:`Detokenization` is just the inverse operation of :term:`tokenization`.
For example, a character tokenizer can detokenize character list into text:

.. code-block:: python

  import lmp.tknzr

  # Create tokenizer instance.
  tokenizer = lmp.tknzr.CharTknzr()

  # Convert token list into text.
  assert tokenizer.dtknz(tks=['a', 'b', 'c']) == 'abc'

Normalize text
--------------
Tokenizer can perform :term:`text normalization`.
For example, when setting ``is_uncased=True``, all uppercase characters will be converted into lowercase characters:

.. code-block:: python

  import lmp.tknzr

  # Create tokenizer instance and setting `is_uncased=True`.
  tokenizer = lmp.tknzr.CharTknzr(is_uncased=True)

  # Normalize text.
  assert tokenizer.norm(txt='ABC') == 'abc'
  # Tokenization normalize text, too.
  assert tokenizer.tknz(txt='ABC') == ['a', 'b', 'c']

Build vocabulary
----------------
A tokenizer's :term:`vocabulary` can be built with the :py:meth:`build_vocab` method.
One first choose a dataset, then build vocabulary on top of the dataset.
Vocabulary are saved as :py:class:`dict` structure with keys and values corresponding to :term:`tokens` and :term:`token ids`, respectively.
One can access the vocabulary through the tokenizer instance attribute ``tk2id``.
For example, we can build character-only vocabulary on top of :py:class:`lmp.dset.DemoDset`:

.. code-block:: python

  import lmp.dset
  import lmp.tknzr

  # Create dataset instance.
  dataset = lmp.dset.DemoDset(ver='train')

  # Create tokenizer instance.
  tokenizer = lmp.tknzr.CharTknzr()

  # Build vocabulary on top of a dataset.
  tokenizer.build_vocab(batch_txt=dataset)

  # Access tokenizer's vocabulary.
  assert '1' in tokenizer.tk2id
  print(tokenizer.tk2id['1'])

Tokens having higher occurrence counts will be added to tokenizer's vocabulary first.
Tokens have the same occurrence counts will be added to tokenizer's vocabulary in the order of their appearence.
Not all tokens need to be included in a tokenizer's vocabulary.
The parameter ``max_vocab`` is the maximum number of tokens to be included in a tokenizer's vocabulary.
When setting ``max_vocab=-1``, one can have unlimited number (only limited by the memory size) of tokens in a tokenizer's vocabulary.

.. code-block:: python

  import lmp.tknzr

  # Create tokenizer instance with limited vocabulary size.
  tokenizer = lmp.tknzr.CharTknzr(max_vocab=5)

  # Build vocabulary on top of a dataset.
  tokenizer.build_vocab(batch_txt=[
    'abcdefg',
    'hijklmnop',
    'qrstuvwxyz',
  ])

  # Vocabulary size is limited.
  assert 'a' in tokenizer.tk2id
  assert 'z' not in tokenizer.tk2id

Sometimes there are tokens which occur only one times or a few.
These tokens are usually named entities or typos.
We will not fix typos or handle named entities.
Instead, one can filter out these tokens by deciding the minimum occurrence count for a token to be included in a tokenizer's vocabulary.
The parameter ``min_count`` serve this purpose.
No filtering will be performed when setting ``min_count=0``.

.. code-block:: python

  import lmp.tknzr

  # Create tokenizer instance and filter out tokens with low occurence counts.
  tokenizer = lmp.tknzr.CharTknzr(min_count=2)

  # Build vocabulary on top of a dataset.
  tokenizer.build_vocab(batch_txt=[
    'aaa',
    'bb',
    'c',
  ])

  # Tokens must satisfy minimum occurrence count constraint.
  assert 'a' in tokenizer.tk2id
  assert 'b' in tokenizer.tk2id
  assert 'c' not in tokenizer.tk2id

Language model training formation
---------------------------------
After building :term:`vocabulary`, one can format a given text into :term:`language model` :term:`training` format.
A language model training format is consist of a :term:`BOS` token, followed by the token list and a :term:`EOS` token.
For example, the text ``hello world`` can be format as ``<bos> hello world <eos>``.
If a token is not in tokenizer's vocabulary, then tokenizer will treat such token as :term:`OOV` token and replace OOV token with :term:`UNK` token.
In general, a language model training format looks like follow::

  <bos> token_1 token_2 <unk> ... token_N <eos> <pad> <pad> ... <pad>

One use :py:meth:`enc` to convert text into language model training format and convert each token into :term:`token id` at the same time.
:term:`PAD` tokens only exist when the length of token list is shorter than required.
One can use :py:meth:`pad_to_max` to pad a token list to specific length.

.. code-block:: python

  import lmp.tknzr
  from lmp.vars import BOS_TKID, EOS_TKID, PAD_TKID, UNK_TKID

  tokenizer = lmp.tknzr.CharTknzr(min_count=2)

  # Build vocabulary on top of a dataset.
  tokenizer.build_vocab(batch_txt=[
    'aaa',
    'bb',
    'c',
  ])

  # Language model training formation.
  tkids = tokenizer.enc(txt='abc')
  assert tkids == [
    BOS_TKID,
    tokenizer.tk2id['a'],
    tokenizer.tk2id['b'],
    UNK_TKID,
    EOS_TKID,
  ]
  assert tokenizer.pad_to_max(max_seq_len=6, tkids=tkids) == [
    BOS_TKID,
    tokenizer.tk2id['a'],
    tokenizer.tk2id['b'],
    UNK_TKID,
    EOS_TKID,
    PAD_TKID,
  ]

One use :py:meth:`dec` to convert language model training format back to original text:

.. code-block:: python

  import lmp.tknzr
  from lmp.vars import BOS_TK, BOS_TKID, EOS_TK, EOS_TKID, PAD_TK, PAD_TKID, UNK_TK, UNK_TKID

  tokenizer = lmp.tknzr.CharTknzr(min_count=2)

  # Build vocabulary on top of a dataset.
  tokenizer.build_vocab(batch_txt=[
    'aaa',
    'bb',
    'c',
  ])

  # Convert language model training format back to text.
  assert tokenizer.dec(tkids=[
    BOS_TKID,
    tokenizer.tk2id['a'],
    tokenizer.tk2id['b'],
    UNK_TKID,
    EOS_TKID,
    PAD_TKID,
  ]) == f'{BOS_TK}ab{UNK_TK}{EOS_TK}{PAD_TK}'

All available tokenizers
------------------------
.. toctree::
  :glob:
  :maxdepth: 1

  *

.. footbibliography::

.. _Python: https://www.python.org/
