Glossary
========

.. glossary::

    detokenize
    detokenization
        Converts list of tokens back to one and only one text.

        For example, when we detokenize ``['a', 'b', 'c']`` based on
        **character**, we get ``'abc'``;
        When we detokenize ``['a', 'b', 'c']`` base on **whitespace**, we get
        ``'a b c'``.

        Detokenization is just the oppsite operation of :term:`tokenization`,
        and detokenization usually don't involve any statistics.

    experiment
        May refer to :term:`tokenizer` training experiment or model training
        experiment.
        One usually train a tokenizer first and then train a model.

    experiment name
        Name of a particular :term:`experiment`.

    experiment path
        If :term:`experiment name` is ``my_exp``, then experiment path is
        ``exp/my_exp``.
        All :term:`experiment` related files will be put under directory ``exp``.

    NFKC
        **Unicode normalization** is a process which convert full-width
        character into half-width, convert same glyph into same unicode, etc.
        It is a standard tool to preprocess text.

        See https://en.wikipedia.org/wiki/Unicode_equivalence for more detail.

    OOV
    out-of-vocabulary
        Refers to :term:`tokens` which are **not** in :term:`vocabulary`.

    token
    tokens
    tokenize
    tokenization
        Chunks text into small pieces (which are called **tokens**).

        For example, when we tokenize text ``'abc 123'`` based on
        **character**, we get ``['a', 'b', 'c', ' ', '1', '2', '3']``;
        When we tokenize text ``'abc 123'`` base on **whitespace**, we get
        ``['abc', '123']``.

        When processing text, one usually need a :term:`tokenizer` to convert
        bunch of long text (maybe a sentence, a paragraph, a document or whole
        bunch of documents) into smaller tokens (may be characters, words,
        etc.) and thus acquire statistic information (count tokens frequency,
        plot tokens distribution, etc.) to perform furthur analyzations.

        How to tokenize is a research problem, and there are many
        statistic-based tokenization models (which we call them
        :term:`tokenizer`) have been proposed.
        One such famous example is STANZA_ proposed by Stanford.

        .. _STANZA: https://stanfordnlp.github.io/stanza/tokenize.html

    Tokenizer
    tokenizer
        Tools for text :term:`tokenization`.
        It can refer to statistic-based tokenization models.

    Vocabulary
    vocabulary
        When processing text, one have to choose how many :term:`tokens` need
        to be analyzed since we have limited memory size. Those chosen tokens
        are referred as **known tokens**, and are collectivly called
        **vocabulary**. For the rest of the tokens (there are a lot of such
        tokens out there) not in the vocabulary are thus called
        :term:`out-of-vocabulary` tokens.
