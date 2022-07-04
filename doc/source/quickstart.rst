Quick Start
===========

We provide installation instructions only for Ubuntu ``20.04+``.

Environment Prerequests
-----------------------
1. We use Python_ with version ``3.8+``.
   You can install Python_ with

   .. code-block:: shell

     apt install python3.8 python3.8-dev

   .. note::

     Currently (2022) the latest version of Python_ supported by PyTorch_ is ``3.8``.
     That's why we install ``python3.8`` instead of ``python3.10``.
     You might need to use ``sudo`` to perform installation.

2. We use PyTorch_ with version ``1.10+`` and CUDA_ with version ``11.2+``.
   This only work if you have **Nvidia** GPUs.
   You can install CUDA_ library with

   .. code-block:: shell

     apt install nvidia-driver-460

   .. note::

     You might need to use ``sudo`` to perform installation.

3. We use pipenv_ to install Python_ dependencies.
   You can install ``pipenv`` with

   .. code-block:: shell

     pip install pipenv

   .. warning::

     Do not use ``apt`` to intall pipenv_.

   .. note::

     You might want to set environment variable ``PIPENV_VENV_IN_PROJECT=1`` to make virtual environment folders
     always located in your Python_ projects.
     See pipenv_ document for details.

Installation
------------
1. Clone the project_ from GitHub.

   .. code-block:: shell

     git clone https://github.com/ProFatXuanAll/language-model-playground.git

2. Change current directory to ``language-model-playground``.

   .. code-block:: shell

     cd language-model-playground

3. Use pipenv_ to create Python_ virtual environment and install dependencies in Python_ virtual environment.

   .. code-block:: shell

     pipenv install

4. Launch Python_ virtual environment created by pipenv_.

   .. code-block:: shell

     pipenv shell

5. Now you can run any scripts provided by this project!
   For example, you can take a look at chinese poem dataset by running :py:mod:`lmp.script.sample_dset`

   .. code-block:: shell

     python -m lmp.script.sample_dset chinese-poem

Training Language Model Pipline
-------------------------------
We now demonstrate a typical :term:`language model` training pipline.

.. note::

   Throughout this tutorial you might see the symbol ``\`` appear several times.
   ``\`` are only used to format our CLI codes to avoid lenthy lines.
   All CLI codes can in practice be fit into one line, but that would make your codes unreadable and should be
   considered as bad choices.

1. Choose a Dataset
~~~~~~~~~~~~~~~~~~~
Choose a dataset to train.

In this example we use :py:class:`lmp.dset.WikiText2Dset` as our demo dataset.

.. seealso::

  :doc:`lmp.dset </dset/index>`
    All available datasets.

2. Choose a Tokenizer and Train it
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Choose your :term:`tokenizer` and train your tokenizer on a dataset.

The following example use whitespace tokenizer :py:class:`lmp.tknzr.WsTknzr` to train on
:py:class:`lmp.dset.WikiText2Dset` dataset since samples in :py:class:`lmp.dset.WikiText2Dset` are English and thus
tokens (words) are separated by whitespace.

We use :py:mod:`lmp.script.train_tknzr` to train our whitespace tokenizer:

.. code-block:: shell

  python -m lmp.script.train_tknzr whitespace \
    --dset_name wiki-text-2 \
    --exp_name my_tknzr_exp \
    --is_uncased \
    --max_vocab -1 \
    --min_count 10 \
    --ver train

We pass ``whitespace`` as the first argument to specify that we will use :py:class:`lmp.tknzr.WsTknzr` as our
tokenizer, and we train our tokenizer on :py:class:`lmp.dset.WikiText2Dset` dataset using ``--dset_name wiki-text-2``
arguments.
We use ``--ver train`` since our :term:`language model` will be trained on the same training version of
:py:class:`lmp.dset.WikiText2Dset` dataset.

We use ``--max_vocab -1`` to include all tokens in Wiki-text-2.
This results in :term:`vocabulary` size around ``30000``, which is a little bit too much.
Thus we use ``--min_count 10`` in conjunction to filter out tokens with occurrence counts less than ``10``.
Here our assumption is that tokens occur less than ``10`` times are likely to be typos, or name entities, or something
else that we believe are not useful.
We use ``--is_uncased`` to convert uppercase letters into lowercase which helps on reducing vocabulary size.
(for example, ``You`` and ``you`` are now treated as same words.)

All arguments we used are just a mather of choice of pre-processing.
You can change them to any values you think the best.

.. seealso::

  :doc:`lmp.tknzr </tknzr/index>`
    All available tokenizers.

3. Evaluate Tokenizer Training Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Following the previous step, you can now use your previously trained (or pre-trained) :term:`tokenizer` to perform
:term:`tokenization` on arbitrary text you want.

In the following example it tokenize the sentence ``hello world`` into string list ``['hello', 'world']``:

.. code-block:: shell

  python -m lmp.script.tknz_txt \
    --exp_name my_tknzr_exp \
    --txt "hello world"

4. Choose a Language Model and Train it
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Now we can train our :term:`language model` with the help of pre-trained :term:`tokenizer`.

This example use :py:mod:`lmp.model.LSTM2000` as our training language model:

.. code-block:: shell

  python -m lmp.script.train_model LSTM-2000 \
    --batch_size 32 \
    --beta1 0.9 \
    --beta2 0.99 \
    --ckpt_step 500 \
    --d_blk 64 \
    --d_emb 100 \
    --dset_name wiki-text-2 \
    --eps 1e-8 \
    --exp_name my_model_exp \
    --log_step 200 \
    --lr 1e-4 \
    --max_norm 1 \
    --max_seq_len 128 \
    --n_blk 8 \
    --n_epoch 10 \
    --p_emb 0.1 \
    --p_hid 0.1 \
    --tknzr_exp_name my_tknzr_exp \
    --ver train \
    --warmup_step 1000 \
    --wd 1e-2

We pass ``LSTM-2000`` as the first argument to specify that we will use :py:class:`lmp.model.LSTM2000` as our language
model, and we train our model on :py:class:`lmp.dset.WikiText2Dset` dataset using ``--dset_name wiki-text-2``
arguments.
We use ``--ver train`` just as we did to our tokenizer.

We will train on Wiki-text-2 dataset for ``10`` epochs, which means our model will be trained on the same samples for
``10`` times.
(This is specified in ``--n_epoch 10``.)
We group samples with ``32`` samples in each group and we fed groups to model one-by-one.
(This is specified in ``--batch_size 32``.)
We call each group a **mini-batch**.
Samples in mini-batch are randomly grouped together in each training epoch, and the order of feeding mini-batches to
model is randomly purmuted.
Thus for ``10`` epochs we might have ``10`` different mini-batches training order and hundreds of thousands of
different mini-batches.

Samples in mini-batch are first pre-processed by our pre-train :term:`tokenizer` (as specified in
``--tknzr_exp_name my_tknzr_exp``), then the processed results are fed into model.
To use a different tokenizer, you can go back to the previous step to see how you can obtain a pre-trained tokenizer.

The training script will save our model training results.  Saving will be triggered every ``500`` updates (as specified
in ``--ckpt_step``).
We call these saved files as :term:`checkpoints`.
In the next step we will use these model checkpoints to perform evaluation.
Checkpoint files are named with the format ``model-\d+.pt``, where ``\d+`` is a integer representing the checkpoint
saving step.
(For example we might save a checkpoint at step ``5000``, and we would have a file with name ``model-5000.pt``.)

The training script will log model training performance, i.e., the output of a **loss function**.
Log will be triggered every ``200`` updates (as specified in ``--log_step``).
You can see the performance logs on CLI.
You can also use browser to see your performance logs using the following script:

.. code-block:: shell

  pipenv run tensorboard

After launch the command, you can open your **browser** with URL http://localhost:6006/ to see your performance logs.

We split the rest of arguments into two groups:

- :term:`Optimization` hyperparameters.
- Regularization tricks for optimization.
- **Model architecture** hyperparameters.

For **optimization**, we use :py:class:`torch.optim.AdamW` as our optimization algorithm.
After performing :term:`gradient descent` on :term:`language model`, we use :py:class:`torch.optim.AdamW` to update our
models' parameters.
The goal of optimization is to maximize the next token prediction log-likelihood, or equivalently to minimize token
prediction negative log-likelihood, or simply cross-entropy.
See :py:class:`torch.nn.CrossEntropyLoss` for details.
Arguments including ``--beta1``, ``--beta2``, ``--eps``, ``--lr`` and ``--wd`` are directly passed to
:py:class:`torch.optim.AdamW`.

For **regularization tricks**, one usually incorporate them to prevent irregular behaviors of model optimization.
One of the tricks we used is called gradient clipping, which is used to avoid gradient become to large (in the sense of
norm) which make parameters value become extremely positive or negative.
Argument ``--max_norm`` is served as gradient clipping boundary.

For **model architecture**, you can simply check a model's constructor (for example,
:py:meth:`lmp.model.LSTM1997.__init__`) to see what parameters are passed to model.
Or you can use ``python -m lmp.script.train_model model_name -h`` to see required arguments on CLI help text.
For the meaning of those model architecture hyperparameters, we recommend you to see models' documents for details.

Just like training :term:`tokenizer`, you can choose any values you think the best.

.. seealso::

  :doc:`lmp.model </model/index>`
    All available language models.

5. Evaluate Language Model Training Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Now we check whether our :term:`language model` is successfully trained.

The following example use :py:class:`lmp.dset.WikiText2Dset` dataset to perform evaluation.
First we check whether our model is **underfitting** or not by running evalution on the training set.

.. code-block:: shell

  python -m lmp.script.eval_dset_ppl wiki-text-2 \
    --batch_size 32 \
    --first_ckpt 0 \
    --exp_name my_model_exp \
    --ver train

The script above will evaluate all :term:`checkpoints` we saved during training (start from ``0`` to last).
We use :term:`perplexity` as evaluation metric.
See :py:meth:`lmp.util.metric.ppl` for perplexity calculation details.

Like model training script, you can use the following script and then use browser to open URL http://localhost:6006/ to
see performance logs for evaluation:

.. code-block::

  pipenv run tensorboard

In general, perplexity is the lower the better.
If you don't see perplexity goes down, then your model is **underfitting**.
You should go back to the previous step to re-train your language model.
Try using different batch size, number of epochs, and all sorts of hyperparameters combination.

If you see perplexity goes down, that is good!
But how low should the perplexity be?
Typically perplexity lower than ``100`` is a good sign of well-trained language models.
We recommed you to see papers paired with the dataset.

We now check whether our model is **overfitting** or not by running evaluation on validation set.

.. code-block:: shell

  python -m lmp.script.eval_dset_ppl wiki-text-2 \
    --batch_size 32 \
    --first_ckpt 0 \
    --exp_name my_model_exp \
    --ver valid

If perplexity on validation set does not do well, then its a sign of overfitting, which means our model do not
generalize outside the training set.
We should go back to re-train our model, then validate again.
If out model still overfitting, then we will re-train again and validate again, and so on.
This process goes on until we reach a point where we get good perplexity on both training and validation dataset.
This means we might have a language model which is able to generalize on dataset we have never used to train
(validation set in this case).
To further verify our hypothesis, we can use another dataset check our model's performance.

.. code-block:: shell

  python -m lmp.script.eval_dset_ppl wiki-text-2 \
    --batch_size 32 \
    --first_ckpt 0 \
    --exp_name my_model_exp \
    --ver test

6. Generate Continual Text
~~~~~~~~~~~~~~~~~~~~~~~~~~
Now we can use our well-trained :term:`language model` to generate continual text given some text segment.
For example:

.. code-block:: shell

  python -m lmp.script.gen_txt top-1 \
    --ckpt 5000 \
    --exp_name my_model_exp \
    --max_seq_len 128 \
    --txt "We are"

We use ``top-1`` to specify we want to use :py:class:`lmp.infer.Top1Infer` as inference method to generate continual
text.
We pass ``"We are"`` as conditional text segment to let model generate continual text.

You can use different :term:`checkpoint` by changing the ``--ckpt 5000`` argument.
All available checkpoints is under the :term:`experiment path` ``exp/my_model_exp``.

.. seealso::

  :doc:`lmp.infer </infer/index>`
    All available inference methods.

7. Record Experiment Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Now you have finished your experiments, you can record your results and compare results done by others.
See :doc:`Experiment Results <experiment/index>` for others' experiment and record yours!

Documents
---------
You can read documents on `this website`_ or use the following steps to build documents locally.
We use Sphinx_ to build our documents.

1. Install documentation dependencies.

   .. code-block:: shell

     pipenv install --dev

2. Build documents.

   .. code-block:: shell

     pipenv run doc

3. Open the root document in your browser.

   .. code-block:: shell

     xdg-open doc/build/index.html


Testing
-------

This is for developer only.

1. Install testing dependencies.

   .. code-block:: shell

     pipenv install --dev

2. Run test.

   .. code-block:: shell

     pipenv run test

3. Get test coverage report.

   .. code-block:: shell

     pipenv run test-coverage

.. _PyTorch: https://pytorch.org/
.. _Python: https://www.python.org/
.. _CUDA: https://developer.nvidia.com/cuda-toolkit/
.. _pipenv: https://pipenv.pypa.io/en/latest/
.. _project: https://github.com/ProFatXuanAll/language-model-playground.git
.. _Sphinx: https://www.sphinx-doc.org/en/master/
.. _`this website`: https://language-model-playground.readthedocs.io/en/latest/index.html
