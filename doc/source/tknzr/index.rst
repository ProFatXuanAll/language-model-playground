Tokenizers
==========

Overview
--------
In this project, a :term:`tokenizer` is a collection of text preprocessing tools including text :term:`tokenization`,
:term:`text normalization` and :term:`language model` :term:`training` formation.
To obtain a tokenizer, one must first create an empty tokenizer, then pick a :term:`dataset` to train an empty tokenizer.

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

Create tokenizer instance
-------------------------
After importing :py:mod:`lmp.tknzr`, one can create :term:`tokenizer` instance through the class attributes of
:py:mod:`lmp.tknzr`.
For example, one can create character tokenizer :py:class:`lmp.tknzr.CharTknzr` and whitespace tokenizer
:py:class:`lmp.tknzr.WsTknzr` as follow:

.. code-block:: python

  import lmp.tknzr

  # Create character tokenizer instance.
  character_tokenizer = lmp.tknzr.CharTknzr(
    is_uncased=False,
    max_vocab=-1,
    min_count=0,
  )

  # Create whitespace tokenizer instance.
  whitespace_tokenizer = lmp.tknzr.WsTknzr(
    is_uncased=False,
    max_vocab=-1,
    min_count=0,
  )

Tokenize text
-------------
A :term:`tokenizer` instance can, of course, tokenize text.
For example, a character tokenizer can tokenize text into a list of character tokens, and a whitespace tokenizer can
tokenize text into a whitespace-separated tokens:

.. code-block:: python

  import lmp.tknzr

  character_tokenizer = lmp.tknzr.CharTknzr(
    is_uncased=False,
    max_vocab=-1,
    min_count=0,
  )
  whitespace_tokenizer = lmp.tknzr.CharTknzr(
    is_uncased=False,
    max_vocab=-1,
    min_count=0,
  )

  # Tokenize text into token list.
  assert character_tokenizer.tknz(txt='abc') == ['a', 'b', 'c']
  assert whitespace_tokenizer.tknz(txt='a b c') == ['a', 'b', 'c']

Each :term:`special token` will be treated as a unit no matter which tokenizer is used.

.. code-block:: python

  import lmp.tknzr
  from lmp.tknzr._base import BOS_TK, EOS_TK

  tokenizer = lmp.tknzr.CharTknzr(
    is_uncased=False,
    max_vocab=-1,
    min_count=0,
  )

  # Tokenize text into token list.
  assert tokenizer.tknz(txt=f'{BOS_TK}abc{EOS_TK}') == [BOS_TK, 'a', 'b', 'c', EOS_TK]

Detokenize token list
---------------------
:term:`Detokenization` is just the inverse operation of :term:`tokenization`.
For example, a character tokenizer can detokenize character list into text:

.. code-block:: python

  import lmp.tknzr

  tokenizer = lmp.tknzr.CharTknzr(
    is_uncased=False,
    max_vocab=-1,
    min_count=0,
  )

  # Convert token list into text.
  assert tokenizer.dtknz(tks=['a', 'b', 'c']) == 'abc'

Normalize text
--------------
Tokenizer can perform :term:`text normalization`.
For example, when set ``is_uncased=True``, all uppercase characters will be converted into lowercase characters:

.. code-block:: python

  import lmp.tknzr

  tokenizer = lmp.tknzr.CharTknzr(
    is_uncased=True,
    max_vocab=-1,
    min_count=0,
  )

  # Normalize text.
  assert tokenizer.norm(txt='ABC') == 'abc'
  # Tokenization normalize text, too.
  assert tokenizer.tknz(txt='ABC') == ['a', 'b', 'c']

Build vocabulary
----------------
A tokenizer's :term:`vocabulary` can be built with the :py:meth:`build_vocab` method.
One first choose a dataset then build vocabulary on top of the dataset.
Vocabulary are saved as :py:class:`dict` structure which keys and values are :term:`tokens` and :term:`token ids`.
One can access the vocabulary through the tokenizer instance attribute ``tk2id``.
For example, we can build character-only vocabulary on top of :py:class:`lmp.dset.DemoDset`:

.. code-block:: python

  import lmp.dset
  import lmp.tknzr

  dataset = lmp.dset.DemoDset(ver='train')
  tokenizer = lmp.tknzr.CharTknzr(
    is_uncased=True,
    max_vocab=-1,
    min_count=0,
  )

  # Build vocabulary on top of a dataset.
  tokenizer.build_vocab(batch_txt=dataset)

  # Access tokenizer's vocabulary.
  assert '1' in tokenizer.tk2id
  print(tokenizer.tk2id['1'])

One can decide how many tokens to include in a tokenizer's vocabulary.
The parameter ``max_vocab`` is the maximum number of tokens to be included in a tokenizer's vocabulary.
When setting ``max_vocab=-1``, one can have unlimited number (of course limited by the memory size) of tokens in a
tokenizer's vocabulary.

.. code-block:: python

  import lmp.tknzr

  tokenizer = lmp.tknzr.CharTknzr(
    is_uncased=True,
    max_vocab=5,
    min_count=0,
  )

  # Build vocabulary on top of a dataset.
  tokenizer.build_vocab(batch_txt=[
    'abcdefg',
    'hijklmnop',
    'qrstuvwxyz',
  ])

  # Vocabulary size is limited.
  assert 'z' not in tokenizer.tk2id

Sometimes there are tokens which occur only one times or a few.
These tokens are usually named entities or even worse, typos.
One can filter out these tokens by deciding the minimum occurrence count for a token to be included in a tokenizer's
vocabulary.
The parameter ``min_count`` serve this purpose.
When setting ``min_count=0`` no filtering will be performed.

.. code-block:: python

  import lmp.tknzr

  tokenizer = lmp.tknzr.CharTknzr(
    is_uncased=True,
    max_vocab=-1,
    min_count=2,
  )

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
For example, the text ``hello world`` can be format as ``[BOS] hello world [EOS]``.
If there are tokens not in tokenizer's vocabulary, then tokenizer will treat these tokens as :term:`OOV` tokens and
replace them with :term:`UNK` token.
In general, a language model training format looks like follow::

  [BOS] token_1 token_2 [UNK] ... token_N [EOS] [PAD] [PAD] ... [PAD]

:term:`PAD` tokens only exist when the length of token list is shorter than required.
One can use ``max_seq_len`` parameter to specify required length.
For example, one can use :py:meth:`enc` to convert text into language model training format and convert each token into
:term:`token id` at the same time.

.. code-block:: python

  import lmp.tknzr
  from lmp.tknzr._base import BOS_TKID, EOS_TKID, PAD_TKID, UNK_TKID

  tokenizer = lmp.tknzr.CharTknzr(
    is_uncased=True,
    max_vocab=-1,
    min_count=2,
  )

  # Build vocabulary on top of a dataset.
  tokenizer.build_vocab(batch_txt=[
    'aaa',
    'bb',
    'c',
  ])

  # Language model training formation.
  assert tokenizer.enc(txt='abc', max_seq_len=6) == [
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
  from lmp.tknzr._base import BOS_TK, BOS_TKID, EOS_TK, EOS_TKID, PAD_TK, PAD_TKID, UNK_TK, UNK_TKID

  tokenizer = lmp.tknzr.CharTknzr(
    is_uncased=True,
    max_vocab=-1,
    min_count=2,
  )

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

When dealing a :term:`mini-batch` of text :term:`samples`, one have to first loop through each text sample to get the
maximum length in a mini-batch, then loop again to encode each text sample.
Fortunately, there is a batch version of encoding method :py:meth:`batch_enc`.
And similarly a batch version of decoding method :py:meth:`batch_dec`.

.. code-block:: python

  import lmp.dset
  import lmp.tknzr

  tokenizer = lmp.tknzr.CharTknzr(
    is_uncased=True,
    max_vocab=-1,
    min_count=2,
  )

  # Build vocabulary on top of a dataset.
  dataset = lmp.dset.DemoDset(ver='train')
  tokenizer.build_vocab(batch_txt=dataset)

  batch_size = 32
  mini_batch = [dataset[index] for index in range(batch_size)]

  # Batch processing.
  print(tokenizer.batch_enc(batch_txt=mini_batch, max_seq_len=33))

All available tokenizers
------------------------
.. toctree::
  :glob:
  :maxdepth: 1

  *

.. _Python: https://www.python.org/
