Glossary
========

.. glossary::

   checkpoint
   checkpoints
     In the process of training :term:`language models`, we need to save our training results (model parameters) for
     later evaluation.  We don't want to save our training results only after training.  We want to save our training
     results every certain amount of update times.  The :term:`step` number triggering save process is called
     **checkpoint**.  All checkpoints will be saved at your :term:`experiment path` and named with format
     ``model-\d+.pt``, where ``\d+`` means checkpoint step.

   detokenize
   detokenization
     Converts list of tokens back to one and only one text.

     For example, when we detokenize ``['a', 'b', 'c']`` based on **character**, we get ``'abc'``;  When we detokenize
     ``['a', 'b', 'c']`` base on **whitespace**, we get ``'a b c'``.

     Detokenization is just the oppsite operation of :term:`tokenization`, and detokenization usually don't involve any
     statistics.

   experiment
     May refer to :term:`tokenizer` training experiment or model training experiment.  One usually train a tokenizer
     first and then train a model.

   experiment name
     Name of a particular :term:`experiment`.

   experiment path
     If :term:`experiment name` is ``my_exp``, then experiment path is ``exp/my_exp``.  All :term:`experiment` related
     files will be put under directory ``exp``.

   language model
   language models
     A language model is a model which can calculate the probability of a given text is comming from human language.

     For example, the text "how are you?" is used in daily conversation and thus language model should output high
     probability (or equivalently low :term:`perplexity`).  On the other hand the text "you are how?" is meaningless
     and thus language model should output low probability (or equivalently high :term:`perplexity`).

     More precisely, language model is an probabilistic algorithm which input is text and output is probability (or
     :term:`perplexity`).  We denote language model as :math:`M` and input text as :math:`x`.  The hypothesis (expected
     behavior) of language models are:

     - If :math:`M(x) \approx 1`, then :math:`x` is very likely comming from human language.
     - If :math:`M(x) \approx 0`, then :math:`x` is not likely comming from human language.

     The common way to evalute a language model is using :term:`perplexity`.  In early days language model are used to
     evaluate generated text from speech recognition.  More recently, language models like GPT_ and BERT_ have shown to
     be useful for lots of downstream NLP tasks including Natural Lanugage Understanding (NLU), Natural Language
     Generation (NLG), Question Answering (QA), cloze test, etc.

    .. _GPT: https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/
       language_understanding_paper.pdf
    .. _BERT: https://arxiv.org/abs/1810.04805

    In this project we have provided script for training language model (:py:mod:`lmp.script.train_model`), evaluating
    language model (:py:mod:`lmp.script.evaluate_model_on_dataset`) and generate text using language model
    (:py:mod:`lmp.script.generate_text`).

    .. seealso::
       lmp.script
         All available scripts related to language model.
       lmp.model
         All available language model.

   NN
   neural network
     In this project we use famous deep learning framework PyTorch_ to implement our language models.

     .. _PyTorch: https://pytorch.org/

     .. seealso::

        lmp.model
          All available models.

   NFKC
     **Unicode normalization** is a process which convert full-width character into half-width, convert same glyph into
     same unicode, etc.  It is a standard tool to preprocess text.

     See https://en.wikipedia.org/wiki/Unicode_equivalence for more detail.

   OOV
   out-of-vocabulary
     Refers to :term:`tokens` which are **not** in :term:`vocabulary`.

   Optimization
   optimization
   gradient descent
     In the context of :term:`neural network` optimization we usually mean to perform **gradient descent** on
     :term:`neural network`.  To perform gradient descent, model need to first perform **forward pass**.  During
     forward pass, model will take a input which we called **tensors** and pass tensors to deeper layers in model for
     calculation.  Every path **tensor** flow throught the model will be recorded and construct a **tensor flowing
     graph**.  The output of forward pass is then used to calculate **loss** on **objective function** (or **loss
     function**).  We can say "we are optimizing our model on objective function by minimizing loss."  We can calculate
     gradient on loss with respect to model output.  Then we can use gradient from loss to perform **back-propagation**
     with the aid of tensor flowing graph.  After back-propagation, all parameters in model get their own gradients,
     then we can do **gradient descent**.

   perplexity
     Perplexity is a way to evaluate :term:`language model`.  Given a text :math:`x` consist of :math:`n` tokens
     :math:`x_1, x_2, \dots, x_n`, we want to calculate the probability of text :math:`x` is comming from human
     language:

     .. math::

        \begin{align*}
        ppl(x) &= \sqrt[n]{\frac{1}{P(x_1, x_2, \dots, x_n)}} \\
        &= \bigg(P(x_1, x_2, \dots, x_n)\bigg)^{\frac{-1}{n}} \\
        &= \bigg(P(x_1) P(x_2|x_1) P(x_3|x_1, x_2) \dots P(x_n|x_1, x_2, \dots, x_{n - 1})\bigg)^{\frac{-1}{n}} \\
        &= \bigg(\prod_{i = 1}^n P(x_i|x_1, \dots, x_{i - 1})\bigg)^{\frac{-1}{n}} \\
        &= e^{\log \prod_{i = 1}^n \big(P(x_i|x_1, \dots, x_{i - 1})\big)^{\frac{-1}{n}}} \\
        &= e^{\frac{-1}{n}\log \prod_{i = 1}^n P(x_i|x_1, \dots, x_{i - 1})} \\
        &= e^{\frac{-1}{n} \sum_{i = 1}^n \log P(x_i|x_1, \dots, x_{i - 1})} \\
        &= \exp\bigg(\frac{-1}{n} \sum_{i = 1}^n \log P(x_i|x_1, \dots, x_{i - 1})\bigg)
        \end{align*}

   step
     Refers to number of times a :term:`language model` has been updated.

   token
   tokens
   tokenize
   tokenization
     Chunks text into small pieces (which are called **tokens**).

     For example, when we tokenize text ``'abc 123'`` based on **character**, we get
     ``['a', 'b', 'c', ' ', '1', '2', '3']``;  When we tokenize text ``'abc 123'`` base on **whitespace**, we get
     ``['abc', '123']``.

     When processing text, one usually need a :term:`tokenizer` to convert bunch of long text (maybe a sentence, a
     paragraph, a document or whole bunch of documents) into smaller tokens (may be characters, words, etc.) and thus
     acquire statistic information (count tokens frequency, plot tokens distribution, etc.) to perform furthur
     analyzations.

     How to tokenize is a research problem, and there are many statistic-based tokenization models (which we call them
     :term:`tokenizer`) have been proposed.  One such famous example is STANZA_ proposed by Stanford.

     .. _STANZA: https://stanfordnlp.github.io/stanza/tokenize.html

   token id
     Since :term:`token` (a string) cannot be directly used to compute, we assign each token a **id** and replace
     tokens with their own ids to perform furthur calculation.  Sometimes we also need a mechaism to convert token id
     back to their original token, in such cases we should assume that the :term:`vocabulary` only consist of
     **unique** token and id pairs.

     For example, we can use a token id to perform embedding matrix lookup, the lookup result is a vector (which we
     suppose to) represent that token.

   Tokenizer
   tokenizer
   tokenizers
     Tools for text :term:`tokenization`.  It can refer to statistic-based tokenization models.

   Vocabulary
   vocabulary
     When processing text, one have to choose how many :term:`tokens` need to be analyzed since we have limited memory
     size.  Those chosen tokens are referred as **known tokens**, and are collectivly called **vocabulary**.  For the
     rest of the tokens (there are a lot of such tokens out there) not in the vocabulary are thus called
     :term:`out-of-vocabulary` tokens.
