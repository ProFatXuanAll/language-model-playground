Scripts
=======

Overview
--------
In this project, we provide a series of scripts for one to :term:`train` a :term:`language model`.
Scripts are categorized into three groups: :term:`dataset`-related, :term:`tokenizer`-related and
:term:`language model`-related scripts.
Dataset-related group has only one script :doc:`lmp.script.sample_dset </script/sample_dset>`.
Tokenizer-related group has 2 scripts, :doc:`lmp.script.train_tknzr </script/train_tknzr>` and
:doc:`lmp.script.tknz_txt </script/tknz_txt>`.
The rest scripts belong to language-model-related group.
One should first execute dataset-related script, then tokenizer-related scripts, and finally language-model-related
scripts.

Sample dataset
--------------
One can use :doc:`lmp.script.sample_dset </script/sample_dset>` to get a glimpse of :term:`dataset` :term:`samples`.
For example, one can sample the demo dataset :py:class:`lmp.dset.DemoDset` as follow:

.. code-block:: shell

  python -m lmp.script.sample_dset demo

One can sample different dataset using different dataset names:

.. code-block:: shell

  python -m lmp.script.sample_dset wiki-text-2

Sampling is always done on the default version of a dataset.
To specify different version, one use ``--ver`` arguments with the desired version:

.. code-block:: shell

  python -m lmp.script.sample_dset demo --ver valid

There are many samples in a dataset.
The default sample is the ``0``\th sample in a dataset.
To see different sample, one use ``--idx`` with sample index other than ``0``:

.. code-block:: shell

  python -m lmp.script.sample_dset demo --idx 1 --ver valid

.. seealso::

  :doc:`lmp.dset </dset/index>`
    All available datasets.

Train a tokenizer
-----------------
One can use the script :doc:`lmp.script.train_tknzr </script/train_tknzr>` to create an empty :term:`tokenizer` and
build tokenizer's :term:`vocabulary` on top of a :term:`dataset`.
For example, one can train a whitespace tokenizer :py:class:`lmp.tknzr.WsTknzr` on the training set of the dataset
:py:class:`lmp.dset.WikiText2Dset`:

.. code-block:: shell

  python -m lmp.script.train_tknzr whitespace \
    --dset_name wiki-text-2 \
    --exp_name my_tknzr_exp \
    --max_vocab 10000 \
    --min_count 0 \
    --ver train

In the above example, we use ``whitespace`` as the first argument to specify that we want to train a whitespace
tokenizer.
An empty whitespace tokenizer is created with the arguments ``is_uncased=False``, ``max_vocab=10`` and ``min_count=2``.
We use ``--dset_name wiki-text-2`` and ``--ver train`` to specify that we want to build tokenizer's vocabulary on top
of the training set of wiki-text-2 dataset.
We use the argument ``--exp_name`` to name our tokenizer training :term:`experiment` as ``my_tknzr_exp``.
The tokenizer training results will be saved under the :term:`experiment path` ``project_root/exp/my_tknzr_exp``.

One can decide how many tokens to include in a tokenizer's vocabulary.
The parameter ``--max_vocab`` is the maximum number of tokens to be included in a tokenizer's vocabulary.
When setting ``--max_vocab -1``, one can have unlimited number (of course limited by the memory size) of tokens in a
tokenizer's vocabulary.
The following example results in vocabulary size around ``30000``:

.. code-block:: shell

  python -m lmp.script.train_tknzr whitespace \
    --dset_name wiki-text-2 \
    --exp_name my_tknzr_exp \
    --max_vocab -1 \
    --min_count 0 \
    --ver train

Sometimes there are tokens which occur only one times or a few.
These tokens are usually named entities or even worse, typos.
One can filter out these tokens by deciding the minimum occurrence count for a token to be included in a tokenizer's
vocabulary.
The parameter ``--min_count`` serve this purpose.
When setting ``--min_count 0`` no filtering will be performed.
The following example results in vocabulary size around ``13000``:

.. code-block:: shell

  python -m lmp.script.train_tknzr whitespace \
    --dset_name wiki-text-2 \
    --exp_name my_tknzr_exp \
    --max_vocab -1 \
    --min_count 10 \
    --ver train

Same character :term:`sequence` with different cases (for example, Apple and apple) will be treated as different tokens.
One can make tokenizer case-insensitive using the argument ``--is_uncased``.
The following example results in vocabulary size around ``12000``:

.. code-block:: shell

  python -m lmp.script.train_tknzr whitespace \
    --dset_name wiki-text-2 \
    --exp_name my_tknzr_exp \
    --is_uncased \
    --max_vocab -1 \
    --min_count 10 \
    --ver train

.. seealso::

  :doc:`lmp.tknzr </tknzr/index>`
    All available tokenizers.

Tokenize text
-------------
After training a :term:`tokenizer`, one can use the :term:`pre-trained` tokenizer to tokenize text.
For example, following the examples in the previous section, we can tokenize text ``hello world`` into
``['hello', 'world']``:

.. code-block:: shell

  python -m lmp.script.tknz_txt --exp_name my_tknzr_exp --txt "hello world"

In the above example, we use ``--exp_name my_tknzr_exp`` to load the pre-trained tokenizer.
We provide the argument ``--txt "hello world"`` to tokenize the character sequence ``"hello world"``.

Train a language model
----------------------
One can use a :term:`language model` to generate continual text on the given text.
Before that, one have to first :term:`optimize` a language model's :term:`loss function` on a :term:`dataset`.
To perform optimization, one can use the language model :term:`training` script
:doc:`lmp.script.train_model </script/train_model>`.
For example, we can train a LSTM (2000 version) language model :py:class:`lmp.model.LSTM2000` on the training set of
wiki-text-2 dataset :py:class:`lmp.dset.WikiText2Dset`.

.. code-block:: shell

  python -m lmp.script.train_model LSTM-2000 \
    --batch_size 64 \
    --beta1 0.9 \
    --beta2 0.99 \
    --ckpt_step 1000 \
    --ctx_win 16 \
    --d_blk 64 \
    --d_emb 300 \
    --dset_name wiki-text-2 \
    --eps 1e-8 \
    --exp_name my_model_exp \
    --log_step 200 \
    --lr 1e-4 \
    --max_norm 1 \
    --max_seq_len 700 \
    --n_blk 8 \
    --p_emb 0.1 \
    --p_hid 0.1 \
    --tknzr_exp_name my_tknzr_exp \
    --ver train \
    --total_step 50000 \
    --warmup_step 10000 \
    --wd 1e-2

Language model arguments
~~~~~~~~~~~~~~~~~~~~~~~~
The first argument in the above example is the name of a language model.
Language models have different structure.
One can see a specific model :term:`hyper-parameters` using ``-h`` arguments.
For example, one can see that ``--d_emb``, ``--d_blk``, ``--n_blk``, ``--p_emb`` and ``--p_hid`` are parts of the LSTM
(2000 version) parameters using the following script:

.. code-block:: shell

  python -m lmp.script.train_model LSTM-2000 -h

Text processing arguments
~~~~~~~~~~~~~~~~~~~~~~~~~
Every language model is paired with a :term:`tokenizer`.
The paired tokenizer will share its :term:`vocabulary` with a language model.
In the example above, we set ``--tknzr_exp_name my_tknzr_exp`` to use a :term:`pre-trained` tokenizer with experiment
name ``my_tknzr_exp``.
One usually use a tokenizer trained on the same dataset and the same version.
Thus the above example use ``--dset_name wiki-text-2`` and ``--ver train`` as in the tokenizer training experiment.

Optimization algorithm arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The optimization algorithm is :py:class:`torch.optim.AdamW`.
Due to memory size limit and computation cost, we chunk dataset into :term:`mini-batch` to perform optimization.
The :term:`batch size` is set by the argument ``--batch_size``.
In the example above, we will fetch ``64`` samples from the wiki-text-2 dataset to perform optimization.
We sample dataset without repetitions util every sample has been used to train once.
For the purpose of parallel computation, each sample in a mini-batch will be :term:`padded` to have the same length.
This is done by setting ``--max_seq_len``.
In the example above, a mini-batch will be padded to have length ``700``.
After padding, a mini-batch in the example above will be chunked into smaller :term:`context window` with length ``16``
in each context window.
An optimization :term:`step` is performed on a context window.
The total number of optimization steps is set by ``--total_step``.
No padding tokens will contribute to loss.
One can adjust context window size by changing the value of ``--ctx_win``.
The arguments directly passed to :py:class:`torch.optim.AdamW` are ``--beta1``, ``--beta2``, ``--eps``, ``--lr`` and
``--wd``.
The ``betas`` parameter for :py:class:`torch.optim.AdamW` are split into ``--beta1`` and ``--beta2``.
The ``eps`` for :py:class:`torch.optim.AdamW` is given by ``--eps``.
The ``weight_decay`` parameter is given by ``--wd``.
The learning rate is given by ``--lr``.
Learning rate is scheduled to linearly warmup to the value given by ``--lr`` and linearly decay to ``0`` after reaching
the peak value.
The number of steps to warmup is set by ``--warmup_step``.
To avoid gradient explosion, we use the max norm argument ``--max_norm`` to make the gradient norm of all parameters
have an upper bound.
Gradients with norm larger than max norm will be clipped to max norm.

Logging arguments
~~~~~~~~~~~~~~~~~
For each ``1000`` steps, we will save the :term:`model parameters` under the :term:`experiment path`
``project_root/exp/my_model_exp``.
This is done by setting ``--ckpt_step 1000`` and ``--exp_name my_model_exp``.
Similarly, by setting ``--log_step 200``, we log model performance for each ``200`` steps.
We use tensorboard to log the model performance.
One can launch tensorboard and open browser with URL http://localhost:6006 to see the performance logs.
Use the following script to launch tensorboard:

.. code-block:: shell

  pipenv run tensorboard

.. seealso::

  :doc:`lmp.model </model/index>`
    All available language models.

Evaluate dataset perplexity
---------------------------
To perform :term:`perplexity` evaluation on a :term:`dataset`, one use the evaluation script
:doc:`lmp.script.eval_dset_ppl </script/eval_dset_ppl>`:

.. code-block:: shell

  python -m lmp.script.eval_dset_ppl wiki-text-2 \
    --batch_size 32 \
    --first_ckpt 0 \
    --exp_name my_model_exp \
    --ver valid

One use a dataset's name as first argument.
The specific version of a dataset to evaluate can be set by ``--ver`` argument.
One need to specify which experiment to evaluate using ``--exp_name`` argument.
Since evaluation does not construct tensor graph, one can use larger ``--batch_size`` compare to :term:`training`.
Other settings like :term:`context window` or maximum sequence length will follow the training settings.
Since :term:`pre-trained` :term:`language model` are saved as :term:`model parameters` :term:`checkpoints`, one also
need to specify which checkpoint to evaluate.
But there are lots of checkpoints can be evaluated, thus we provide two arguments ``--first_ckpt`` and ``--last_ckpt``
for one to specify the starting (first) and the end (last) checkpoint numbers to be evaluated.
To evaluate every checkpoints, one can simply set ``--first_ckpt 0`` and ``--last_ckpt -1``.

.. code-block:: shell

  python -m lmp.script.eval_dset_ppl wiki-text-2 \
    --batch_size 32 \
    --first_ckpt 0 \
    --exp_name my_model_exp \
    --last_ckpt -1 \
    --ver valid

We use tensorboard to log the evaluation results.
One can launch tensorboard and open browser with URL http://localhost:6006 to see the evaluation results.
Use the following script to launch tensorboard:

.. code-block:: shell

  pipenv run tensorboard

Evaluate text perplexity
------------------------
To evaluate :term:`language model` :term:`perplexity` on a given text, one do not need to build a :term:`dataset` but
instead use the script :doc:`lmp.script.eval_txt_ppl </script/eval_txt_ppl>`.

.. code-block:: shell

  python -m lmp.script.eval_txt_ppl --ckpt -1 \
    --exp_name my_model_exp \
    --txt "hello world"

We use ``--ckpt`` to specify the evalution checkpoint, and use ``--exp_name`` to specify evaluation
:term:`experiment name`.
In the above example, we evaluate the character sequence ``"hello world"`` by setting ``--txt "hello world"``.

Generate continual text
-----------------------
Finally, one can use a :term:`pre-trained` :term:`language model` to generate continual text on the given text.
One use the script :doc:`lmp.script.gen_txt </script/gen_txt>` and select a decoding strategy to generate continual
text.

.. code-block:: shell

  python -m lmp.script.gen_txt top-P \
    --ckpt -1 \
    --exp_name my_model_exp \
    --max_seq_len 128 \
    --p 0.9 \
    --txt "I 'm on the highway to hell"

The first argument is the name of inference method.
Different inference method have different arguments.
In the example above, the top-P inference method must specify the probability threshold ``--p`` to perform inference.
As in evaluating language model, one specify :term:`experiment name` and :term:`checkpoint` to perform generation.
When setting ``--ckpt -1``, one use the last checkpoint to generate continual text.
The conditional text to perform generation is give by ``--txt``.
To avoid non-stopping generation, one specify maximum length to generate by setting ``--max_seq_len``.

.. seealso::

  :doc:`lmp.infer </infer/index>`
    All available inference methods.

.. toctree::
  :caption: All available scripts are listed below:
  :glob:
  :maxdepth: 1

  sample_dset
  train_tknzr
  tknz_txt
  train_model
  eval_dset_ppl
  eval_txt_ppl
  gen_txt
