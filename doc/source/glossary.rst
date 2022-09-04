Glossary
========

.. glossary::

  back-propagation
    A efficient way to calculate gradient of :term:`loss function` with respect to each :term:`model parameter`.
    See :footcite:`rumelhart1986learning` for algorithm detail.

  batch size
    Number of samples in a :term:`mini-batch`.

  BOS
  BOS token
  BOS tokens
  begin-of-sequence
  begin-of-sequence token
  begin-of-sequence tokens
    **BOS** token is a :term:`special token` which represent the begining of a sequence.
    In this project, BOS token represent the begining of a given text passage.
    BOS token is the first input of a :term:`language model`.
    A language model is :term:`trained` so that when receives a BOS token it must predict the most possible token that can appear at the start of a text passage.

  checkpoint
  checkpoints
    When training :term:`language models`, we save :term:`model parameters` for later evaluation.
    We save model parameters every certain amount of :term:`step`.
    The step number triggering saving process is called **checkpoint**.
    All checkpoints will be saved at your :term:`experiment path` and named with format ``model-\d+.pt``, where ``\d+`` means checkpoint step.

  context window
  context window size
  context windows
    When performing :term:`time-series` tasks, one sometimes need to deals with long :term:`sequences` and cannot fit the whole sequences into memory.
    In this case we usually chuck a time-series data into subsequences (or frames) and train :term:`model` on these subsequences.
    The fancy name for subsequences is **context window**, and the size of each subsequence is called **context window size**.
    When optimize :term:`RNN` models, one usually optimize performance on subsequences instead of the whole sequence to ease the :term:`optimization` problems like :term:`gradient explosion` and :term:`gradient vanishing`.
    Context windows can have overlaps.
    The number of overlapping tokens is called :term:`stride`.
    Both context window size and stride are treated as preprocessing :term:`hyperparameters`.

  cross entropy
  cross entropy loss
  cross-entropy
  cross-entropy loss
    A :term:`loss function` used to :term:`optimize` classifiers.
    Suppose that we are performing a :math:`C` classes classification task and a classifier produce a probability distribution :math:`P(x) = \pa{P_1(x), \dots, P_C(x)}` given a input :math:`x`.
    If the ground truth correspond to :math:`x` is :math:`y` (note that :math:`y \in \set{1, \dots, C}`), then **cross entropy loss** of :math:`(x, y)` is calculated as follow

    .. math::

      \operatorname{CE}(x, y) = -\log P_y(x).

    When :math:`P_y(x) \approx 1`, we have :math:`P_i(x) \approx 0` for every other non-:math:`y`-th class :math:`i`.
    Thus if one use cross entropy to optimize :term:`model`, then one is maximize model's loglikelihood.

  CUDA
    A GPU library developed by Nvidia.

  dataset
    In our project, a **dataset** is consist of text samples.

    .. seealso::

      :doc:`lmp.dset </dset/index>`
        All available dataset.

  detokenize
  detokenization
    **Detokenization** is just the oppsite operation of :term:`tokenization`;
    it converts token list into text.

    For example, when we use character tokenizer to detokenize ``['a', 'b', 'c']`` we get ``'abc'``;
    when we use whitespace tokenizer to detokenize ``['a', 'b', 'c']`` we get ``'a b c'``.

  EOS
  EOS token
  EOS tokens
  end-of-sequence
  end-of-sequence token
  end-of-sequence tokens
    **EOS** token is a :term:`special token` which represent the end of a sequence.
    In this project, EOS token represent the end of a given text passage.
    EOS token is the prediction target of the last input token of a :term:`language model`.
    In this project, any tokens that follows EOS token can only be :term:`PAD` tokens, and language models are not :term:`trained` to produced meaningful output when seeing EOS tokens and PAD tokens.

  epoch
  epochs
    An iteration which loops through each :term:`sample` in a :term:`dataset` is called an **epoch**.

  experiment
    May refer to :term:`tokenizer` training experiment or :term:`language model` training experiment.
    One usually train a tokenizer first and then train a language model.

  experiment name
    Name of a particular :term:`experiment`.

  experiment path
    All :term:`experiment` files are put under directory ``project_root/exp``.
    If :term:`experiment name` is ``my_exp``, then experiment path is ``project_root/exp/my_exp``.

  forward pass
    The process which a :term:`model` takes a input :term:`tensor` and calculates with its :term:`parameters` to achieve certain goal is called **forward pass**.
    In PyTorch_ framework this correspond to :py:meth:`forward()` method of :py:class:`torch.nn.Module`.

  gradient descent
    If we have a :term:`loss function` :math:`L`, then the direction of maximizing :math:`L` with respect to a :term:`model parameter` :math:`W` is :math:`\nabla_W L`, the gradient of :math:`L` with respect to :math:`W`.
    Thus to minimize :math:`L`, one has to go alone the opposite (negative) direction of gradient :math:`\nabla_W L`

    .. math::

      W_{\operatorname{new}} = W_{\operatorname{old}} - \eta \nabla_{W_{\operatorname{old}}} L.

    Where :math:`\eta` is :term:`learning rate`.
    We expect to have :math:`L(W_{\operatorname{new}}) \leq L(W_{\operatorname{old}})`.
    To perform **gradient descent**, model need to first perform :term:`forward pass` to obtain prediction loss.
    Currently the most efficient way to calculate gradients is by the algorithm :term:`back-propagation`.
    After obtaining gradients we can then perform gradient descent.

  gradient explosion
  gradient vanishing
    When perform :term:`gradient descent`, if the calculated gradients have large norm (large in magnitude), then :term:`model parameters` will also have large norm and results in values like Inf or NaN which makes model malfunctioning.
    This is called **gradient explosion**.
    On the other extreme, if the calculated gradients have norm closed to zero, then :term:`model parameters` will be updated extremely slow.
    This is called **gradient vanishing**.
    These two cases happed all the times when :term:`optimize` deep learning :term:`model` by gradient descent, especially when optimizing :term:`RNN` models.

    One can use gradient clipping to enforce the magnitude of gradients fall within certain boundary.
    We use ``--max_norm`` in :doc:`lmp.script.train_model </script/train_model>` to clip gradients.
    Gradient clipping can ease the gradient explosion but not vanishing.
    To solve gradient vanishing, one have to design specific model structure so that gradients of parameters closed to input layer is guarenteed to have almost identical scale.
    For example, the internal states of :py:class:`~lmp.model.LSTM1997` is one such mechanism.
    Other mechanisms like residual connection :footcite:`he2016deep` are also proposed.

  hidden states
  initial hidden states
    When a :term:`model` receives a :term:`time-series` data, some of the early computation results can serve as future input and perform further computation.
    These computation results generated by the model on the fly are called **hidden states**.
    All hidden states for each time step have identical structure.
    This means we can use for-loops to calculate hidden states.
    By the nature of for-loops, we must provide **initial hidden states** to make for-loops work.
    This means initial hidden states may not be generated on the fly but previously defined instead.
    One usually set initial hidden states to zeros.
    One can also let initial hidden states be a part of :term:`model parameters`.
    For simplicity, we set initial hidden states to zeros in this project.

  hyperparameter
  hyperparameters
    A :term:`model` can have the same structure with different number of layers and units.
    The specific number of layers and units are called **hyperparameters**.
    Hyperparameters are decided before training.
    In general, all experiment related parameters are hyperparameters.
    This includes cofiguration for evaluation and inference.

  label smoothing
    When performing classification, one usually optimized classifier model to predict correct label for the corresponding input with high confidence.
    High confidence means that model will output :math:`0` or :math:`1` but nothing else.
    Some argue that optimizing model to have high confidence is hard, but is comparatively easier to optimize for slightly lower confidence.
    Optimizing model with slightly lower confidence is called **label smoothing**.

    Precisely, suppose that we are given an input-ouput pairs :math:`(x, y)` and a possible anser range :math:`C`.
    Suppose also that :math:`P(x) = \pa{P_1(x), \dots, P_C(x)}` is the classification probability distribution given input :math:`x`.
    When optimizing model with high confidence, we are expecting model to output the following probability distribution:

    .. math::

      \forall i \in \set{1, \dots, C}, P_i(x) = \begin{cases}
        1 & \text{if } i = y \\
        0 & \text{otherwise}
      \end{cases}.

    When optimizing model with label smoothing, one expects model to output the following probability distribution:

    .. math::

      \forall i \in \set{1, \dots, C}, P_i(x) = \begin{cases}
        1 - \epsilon            & \text{if } i = y \\
        \dfrac{\epsilon}{C - 1} & \text{otherwise}
      \end{cases}.

    The value :math:`\epsilon` is given as hyperparameter and is typically a small positive number less than :math:`1`.
    Observe that the two formulas above are identical when :math:`\epsilon = 0`.

  language model
  language models
    A **language model** is a :term:`model` which calculates the probability of a given text is comming from human language.
    For example, the text "How are you?" is used in daily conversation and thus language model should output high probability or equivalently low :term:`perplexity`.
    On the other hand, the text "You how are?" is meaningless and thus language model should output low probability or equivalently high perplexity.

    More precisely, language model is an algorithm which inputs text and outputs probability.
    If a language model :math:`M` has :term:`model parameters` :math:`\theta` and takes a input text :math:`x`, then we can interprete :math:`M(x; \theta)` by the following rules

    - If :math:`M(x; \theta) \approx 1`, then :math:`x` is very likely comming from human language.
    - If :math:`M(x; \theta) \approx 0`, then :math:`x` is unlikely comming from human language.

    The usual way to evaluate a language model is :term:`perplexity`.
    In 1990s or earlier, language model are used to evaluate generated text from speech recognition and machine translation.
    More recently (after 2019), language models with huge number of parameters (like GPT :footcite:`radford2018improving` and BERT :footcite:`devlin2019bert`) have been shown to be useful for a lots of downstream NLP tasks, including Natural Language Understanding (NLU), Natural Language Generation (NLG), Question Answering (QA), cloze test, etc.

    In this project we provide scripts for training language model (:doc:`lmp.script.train_model </script/train_model>`), evaluating language model (:doc:`lmp.script.eval_dset_ppl </script/eval_dset_ppl>`) and generating continual text using language model (:doc:`lmp.script.gen_txt </script/gen_txt>`).

    .. seealso::

      :doc:`lmp.script </script/index>`
        All available scripts related to language model.
      :doc:`lmp.model </model/index>`
        All available language model.

  learning rate
    Gradients of loss with respect to :term:`model parameters` is served as the direction of :term:`optimization`.
    But large magnitude of gradients can make optimization hard :footcite:`rumelhart1986learning`.
    Thus one scale down gradients by multiplying a small number called **learning rate**.
    Setting learning rate to small number typically make optimization process longer but stable.
    Setting learning rate to large number typically make optimization process quicker but divergent.
    One rule to keep in mind is that one should use small learning rate when deal with huge number of :term:`model parameters`.

  log path
    All :term:`experiment` log files are put under directory ``project_root/exp/log``.
    If :term:`experiment name` is ``my_exp``, then experiment log path is ``project_root/exp/log/my_exp``.

  loss
  loss function
    A **loss function** is a function which is used to :term:`optimize` and estimate the performance of :term:`model`.
    The input of loss function is consist of :term:`model parameters` and :term:`dataset` :term:`samples`.
    The output of loss function is called **loss**.
    In deep learning field one usually use two different functions for optimization and evaluation.
    For example, we use :term:`cross entropy loss` to optimize :term:`language model` and use :term:`perplexity` to evaluate language model.
    A loss function must have a lower bound so that the optimization process has a chance to approximate the lower bound in finite number of times.
    Without lower bound one cannot know the performance of model by the loss it produces.

  mini-batch
    We split dataset into little :term:`sample` chunks when (:term:`CUDA`) memory cannot fit entire :term:`dataset`.
    Each sample chunk is called a **mini-batch**.
    In deep learning field one usually use mini-batch to perform :term:`optimization` instead of entire dataset.

  model
  model parameter
  model parameters
  parameter
  parameters
    A **model** is an algorithm which takes a input text and performs calculation with certain numbers.
    That certain numbers are called **model parameters** and their values are adjusted by :term:`optimization` process.

    .. seealso::

      :doc:`lmp.model </model/index>`
        All available language models.

  NN
  neural network
    PyTorch_ is a famous deep learning framework that provides lots of **neural network** utilities.
    In this project we use PyTorch to implement :term:`language models`.

  NFKC
    Many unicode characters can represent the same unicode character.
    For example, a unicode character can have full-width (e.g. ``１``) and half-width (e.g. ``1``);
    Japanese puts smaller character after another syllable to make syllable before longer (e.g. ``ｱｲｳｴｵ`` and ``アイウエオ``).
    **Unicode normalization** is a process which maps different representation of a unicode character to the same unicode, and **NFKC** is a way to achieve unicode normalization.
    It is a standard tool to preprocess text.
    See https://en.wikipedia.org/wiki/Unicode_equivalence and https://unicode.org/reports/tr15/ for more details.

  Optimization
  optimization
  Optimize
  optimize
  optimized
  train
  trained
  training
    A process is called **optimization** or **training** if it takes a :term:`model` :math:`M` with :term:`parameter` :math:`\theta` and a :term:`loss function` :math:`L`, continually adjust :math:`\theta` to make :math:`L` closed to its lower bound in a finite number of times.
    In the context of training :term:`neural network`, **optimization** usually means to perform :term:`gradient descent`.

  PAD
  PAD token
  PAD tokens
  padded
  padding
  paddings
  padding token
  padding tokens
    **PAD** token is a :term:`special token` which represent the padding tokens.
    If a :term:`mini-batch` is consist of token :term:`sequences` with different lengths, then such mini-batch will be appended with padding tokens so that token sequence have the same length.
    This is needed since we are perform parallel computation when :term:`training` a :term:`language model`.
    In this project, language models are not trained to produced meaningful output when seeing PAD tokens.

  perplexity
    **Perplexity** is a way to evaluate :term:`language model`.
    Given a text :math:`x` consist of :math:`n` tokens :math:`x = (x_1, x_2, \dots, x_n)`.
    For each :math:`i \in \set{1, \dots, n}`, the probability of next token being :math:`x_i` preceeded by :math:`x_1, \dots, x_{i-1}` is denoted as :math:`P(x_i|x_1, \dots, x_{i-1})`.
    The perplexity of :math:`x`, denoted as :math:`\operatorname{ppl}(x)`, is defined as follow

    .. math::

      \begin{align*}
      \operatorname{ppl}(x) &= \pa{P(x_1, x_2, \dots, x_n)}^{-1/n}                                                                             \\
                            &= \pa{P(x_1) \times P(x_2|x_1) \times P(x_3|x_1, x_2) \times \dots \times P(x_n|x_1, x_2, \dots, x_{n-1})}^{-1/n} \\
                            &= \pa{\prod_{i=1}^n P(x_i|x_1, \dots, x_{i-1})}^{-1/n}                                                            \\
                            &= 2^{\displaystyle \pa{\log_2 \pa{\prod_{i=1}^n P(x_i|x_1, \dots, x_{i-1})}^{-1/n}}}                              \\
                            &= 2^{\displaystyle \pa{\dfrac{-1}{n} \log_2 \pa{\prod_{i=1}^n P(x_i|x_1, \dots, x_{i-1})}}}                       \\
                            &= 2^{\displaystyle \pa{\dfrac{-1}{n} \sum_{i=1}^n \log_2 P(x_i|x_1, \dots, x_{i-1})}}.
      \end{align*}

    If all probabilities :math:`P(x_i|x_1, \dots, x_{i-1})` are high, then perplexity is low.
    Thus we expect a well-trained language model to have low perplexity.

  Pre-trained
  pre-trained
    Abbreviation for "previously trained".

  RNN
  recurrent neural network
    A :term:`neural network` which some of its nodes in later layers connect to nodes in earlier layers.

    .. seealso::

      :doc:`lmp.model </model/index>`
        All available language models.

  sample
  samples
    In our project a sample in a :term:`dataset` is a text (character :term:`sequence`).

  sequence
  sequences
    A data structure which is ordered by integer index.
    We use sequence and :term:`time-series` interchangably in this project.

  Special token
  Special tokens
  special token
  special tokens
    A **special token** is an artifical :term:`token` which is used to perform specific computation.
    In this project, special tokens are added to each :term:`sample` in :term:`dataset` when :term:`training` :term:`language models`.

  step
    Number of times a :term:`language model` has been updated.

  stride
    Number of overlapping tokens between two :term:`context windows`.
    For example, suppose that we set context window size to ``4``, and set stride to ``2``.
    Then the text ``hello world`` will be splited into ``5`` character subsequences as follow:

    .. code-block::

      hell
      llo
      lo w
       wor
      orld

  tensor
  tensors
    A generalized version of matrix is called **tensor**.
    In our scenario we means stacking matrix.
    For example, if we have a list of matrix with shape :math:`(2, 3)` and there are :math:`5` matrices in the list, then we can construct a tensor with shape :math:`(5, 2, 3)` by stacking all :math:`5` matrices together.
    See PyTorch_ tensor :py:class:`torch.Tensor` for more coding example.

  text normalization
    In this project, the term **text normalization** is a three steps process on a given text:

    1. Perform :term:`NFKC` normalization on the given text.
       For example, ``_１__２____３_`` is normalized into ``_1__2____3_``, where ``_`` represents whitespace.
    2. Replace consequtive whitespaces with single whitespace.
       For example, ``_1__2___3_`` will become ``_1_2_3_``, where ``_`` represents whitespace.
    3. Strip (remove) leading and trailing whitespaces.
       For example, ``_1_2_3_`` will become ``1_2_3``, where ``_`` represents whitespace.

    One additional step may be applied depends on how you treat cases.
    If cases do not matter (which is called **case-insensitive**), then text normalization will transform all uppercase characters into lowercase characters.
    For example, ``ABC``, ``AbC``, ``aBc`` will all become ``abc``.
    If case do matter (which is called **case-sensitive**), then no additional steps will to be applied.

  time-series
    A data structure which is ordered by integer index where indices are given the meaning of time.
    Common **time-series** data are sounds and natural languages.
    For example, the sentence "I like to eat apple." can be treated as a character sequence where the first character (correspond to integer index ``0``) is "I", the second character (correspond to integer index ``1``) is whitespace " ", and the last character (correspond to integer ``19``) is ".".
    We use :term:`sequence` and time-series interchangably in this project.

  token
  tokens
  tokenize
  tokenizer
  tokenizers
  tokenization
    Computer treats everything as number.
    To perform text related tasks, one usually chunks text into smaller pieces (called **tokens**) and convert each piece into number so that computer can easily process them.

    For example, when we tokenize text ``'abc 123'`` based on **character**, we get
    ``['a', 'b', 'c', ' ', '1', '2', '3']``;
    When we tokenize text ``'abc 123'`` base on **whitespace**, we get ``['abc', '123']``.

    The tool to chunk text into tokens is called **tokenizer**.
    How to tokenize is a research problem.
    There are many tokenizer have been proposed (e.g. STANZA_, proposed by Stanford).
    In this project our tokenizers provide utilities including tokenization, text normalization and :term:`language model` training formation.

    .. seealso::

      :doc:`lmp.tknzr </tknzr/index>`
        All available tokenizers.

  token id
  token ids
    Since computer only compute numbers and :term:`tokens` are text, we have to assign each token an integer number (called **token id**) and use token ids instead of tokens to perform computation.
    In our project, assigning each token an unique integer is called building :term:`vocabulary`.

  truncate
  truncation
    In this project, this term is used to refer to :term:`truncate` a :term:`token` list into specified length.
    This is the opposite operation of :term:`padding`.

  UNK
  unknown token
  unknown tokens
    **UNK** token is a :term:`special token` which represent the unknown token.
    If :term:`tokenizer` encounter an :term:`out-of-vocabulary` token when convert tokens into :term:`token ids`, tokenizer will treat such token as UNK token and convert it to UNK token id.
    In this project, :term:`language models` are :term:`trained` to produced meaningful output when seeing UNK tokens.
    When encounter a UNK token, language model can only produce next token prediction based on tokens other than UNK.

  Vocabulary
  vocabulary
  OOV
  out-of-vocabulary
    A :term:`language model` is paired with a :term:`tokenizer`.
    How many :term:`tokens` (characters, words, or else) a language model can learn is contrainted by model complexity and memory size.
    A token set learnt by a language model is called **vocabulary**.
    The number of tokens in a vocabulary is called **vocabulary size**.
    Tokens not in the vocabulary of a language model are called :term:`out-of-vocabulary` tokens.

.. footbibliography::

.. _PyTorch: https://pytorch.org/
.. _STANZA: https://stanfordnlp.github.io/stanza/tokenize.html
