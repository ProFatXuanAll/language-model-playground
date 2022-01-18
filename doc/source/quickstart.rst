Quick Start
===========

We provide installation instructions only for Ubuntu ``20.04+``.

Environment Prerequests
-----------------------
1. We use Python_ with version ``3.8+``.  You can install Python_ with

   .. code-block:: shell

      apt install python3.8 python3.8-dev

   .. note::

      Currently the latest version of Python_ supported by PyTorch_ is ``3.8``.  That's why we install ``python3.8``
      instead of ``python3.10``.  You might need to use ``sudo`` to perform installation.

2. We use PyTorch_ with version ``1.10+`` and CUDA_ with version ``11.2+``.  This only work if you have **Nvidia**
   GPUs.  You can install CUDA_ library with

   .. code-block:: shell

      apt install nvidia-driver-460

   .. note::

      You might need to use ``sudo`` to perform installation.

3. We use pipenv_ to install Python_ dependencies.  You can install ``pipenv`` with

   .. code-block:: shell

      pip install pipenv

   .. warning::

      Do not use ``apt`` to intall pipenv_.

   .. note::

      You might want to set environment variable ``PIPENV_VENV_IN_PROJECT=1`` to make virtual environment folders
      always located in your Python_ projects.  See pipenv_ document for details.

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

5. Now you can run any scripts provided by this project!  For example, you can take a look at chinese poem dataset by
   running :py:mod:`lmp.script.sample_dset`

   .. code-block:: shell

      python -m lmp.script.sample_dset chinese-poem

Training Language Model Pipline
-------------------------------
We now demonstrate a basic :term:`language model` training pipline.

.. note::

   Throughout this tutorial you might see the symbol ``\`` appear several times.  ``\`` are only used to format our
   CLI codes to avoid long lines.  All CLI codes should be able to fit-in one line, but this would make your code
   unreadable and should be considered as a bad choice.

1. Choose a Dataset
~~~~~~~~~~~~~~~~~~~
Choose a dataset to train.

In this example we use :py:class:`lmp.dset.WikiText2Dset` as our dataset.

.. seealso::

   :py:mod:`lmp.dset`
     All available dataset.

2. Choose a Tokenizer
~~~~~~~~~~~~~~~~~~~~~
Choose a :term:`tokenizer` and train :term:`tokenizer` on dataset we already choose.

In this example we use :py:class:`lmp.tknzr.WsTknzr` since all samples in :py:class:`lmp.dset.WikiText2Dset` are
whitespace separated.

We use :py:mod:`lmp.script.train_tknzr` to train :term:`tokenizer` given following arguments:

.. code-block:: shell

   python -m lmp.script.train_tknzr whitespace \
     --dset_name wiki-text-2 \
     --exp_name my_tknzr_exp \
     --is_uncased \
     --max_vocab -1 \
     --min_count 10 \
     --ver train

We use ``whitespace`` to specify we want to use :py:class:`lmp.tknzr.WsTknzr` as our :term:`tokenizer`, and we train
our :term:`tokenizer` on Wiki-text-2 dataset using ``--dset_name wiki-text-2`` arguments.  We use ``--ver train`` since
our :term:`language model` will be trained on training version of Wiki-text-2, and we simply treat :term:`OOV` in both
validation and test versions as unknown words.

We use ``--max_vocab -1`` to include all :term:`tokens` in Wiki-text-2.  This results in :term:`vocabulary` size
around ``30000``, which is a little bit too much.  Thus we also use ``--min_count 10`` to filter out all :term:`tokens`
whose frequency are lower than ``10``.  Here we simply assume that all :term:`tokens` occur less than ``10`` times
might be typos, name entities, digits, or something else that we believe are not useful.  We also use ``--is_uncased``
to convert all uppercase letters into lowercase, this also help to reducing :term:`vocabulary` size.  (for example,
``You`` and ``you`` are now treated as same words)

All arguments we used are just a mather of choice for pre-processing.  You can change them to any values you want.

.. seealso::

   :py:mod:`lmp.tknzr`
     All available :term:`tokenizers`.

3. Evaluate Tokenizer
~~~~~~~~~~~~~~~~~~~~~
After training :term:`tokenizer`, you can now use your pre-trained :term:`tokenizer` to :term:`tokenize` arbitrary
text.

For example, you can try to :term:`tokenize` ``hello world`` with script :py:mod:`lmp.script.tknz_txt`:

.. code-block:: shell

   python -m lmp.script.tknz_txt \
     --exp_name my_tknzr_exp \
     --txt "hello world"

You should see something like ``['hello', 'world']``.

4. Choose a Language Model
~~~~~~~~~~~~~~~~~~~~~~~~~~
Now we can train our :term:`language model` with the help of pre-trained :term:`tokenizer`.

In this example we use :py:mod:`lmp.model.LSTM` as our training target.  We use :py:mod:`lmp.script.train_model` to
train :term:`language model` as follow:

.. code-block:: shell

   python -m lmp.script.train_model LSTM \
     --batch_size 32 \
     --beta1 0.9 \
     --beta2 0.99 \
     --ckpt_step 1000 \
     --dset_name wiki-text-2 \
     --eps 1e-8 \
     --exp_name my_model_exp \
     --log_step 200 \
     --lr 1e-4 \
     --max_norm 1 \
     --max_seq_len -1 \
     --n_epoch 10 \
     --tknzr_exp_name my_tknzr_exp \
     --ver train \
     --d_emb 100 \
     --d_hid 300 \
     --n_hid_lyr 2 \
     --n_post_hid_lyr 2 \
     --n_pre_hid_lyr 2 \
     --p_emb 0.1 \
     --p_hid 0.1 \
     --wd 1e-2

:py:mod:`lmp.script.train_model` have similar structure as :py:mod:`lmp.script.train_tknzr`;  We use ``LSTM`` to
specify we want to use :py:class:`lmp.model.LSTMModel` as our :term:`language model`, and train our model on
Wiki-text-2 dataset using ``--dset_name wiki-text-2`` arguments.  We use ``--ver train`` to specify we want to use
training version of Wiki-text-2 which is also used to train our :term:`tokenizer`.

We will train on Wiki-text-2 dataset for ``10`` **epochs**, which means we will repeatly train on sample dataset for
``10`` times.  (This is specified in ``--n_epoch 10``.)  Each time we group all samples in Wiki-text-2 with group size
``32``, and sequentially feed them to model.  (This is specified in ``--batch_size 32``.)  We call one such group as a
**mini-batch**.  All samples in mini-batch are randomly gathered in every epoch, and the order to feed mini-batches to
model are randomly purmuted.  Thus when we train ``10`` epochs we might have ``10`` different mini-batches training
order and hundreds of thousands of different mini-batches.

All samples in mini-batch are first pre-processed by our pre-train :term:`tokenizer` (as specified in
``--tknzr_exp_name my_tknzr_exp``) and then fed into model.  If you think you need a different :term:`tokenizer`, you
can go back to previous step to see how you can obtain a pre-trained :term:`tokenizer`.

We will output our model training result and save them as files (more precisely, compressed pickle files).  Save will
trigger every ``1000`` updates (as specified in ``--ckpt_step``).  We call these saved files as :term:`checkpoint`, all
they saved are model parameters.  Later we will reuse these model parameters to perform further operation such as
:term:`perplexity` evaluation or text generation.  We save these files with name ``model-\d+.pt``, where ``\d+`` means
digits.  (For example we might save at :term:`checkpoint` ``5000`` as ``model-5000.pt``.)

We also log our model performance during training, i.e., **loss function** output.  Log will trigger every ``200``
updates (as specified in ``--log_step``).  You can see performance logs on your CLI, or you can use browser to see your
performance logs by the following script:

.. code-block::

   pipenv run tensorboard

After launch the command, you should open your **browser** and type http://localhost:6006/ to see your performance logs.

For the rest arguments, we split them into two categories:

- :term:`Optimization` hyperparameters.
- **Model architecture** hyperparameters.

For :term:`optimization`, we only provide you with one :term:`optimization` method, namely
:py:class:`torch.optim.Adam`.  We use :py:class:`torch.optim.Adam` to perform :term:`gradient descent` on
:term:`language model`.  Our :term:`optimization` target is to minimize token prediction negative log-likelihood, or
simply cross-entropy.  (This is equivalent to maximize log-likelihood, or just likelihood.)  See
:py:class:`torch.nn.CrossEntropyLoss` for loss function.  Arguments including ``--beta1``, ``--beta2``, ``--eps``,
``--lr`` and ``--wd`` are directly passed to :py:class:`torch.optim.Adam`.

For **model architecture**, you can simply checkt the model's constructor to see what parameters the model needed. Or
you can use ``python -m lmp.script.train_model model_name -h`` to see parameters on CLI.  For the meaning of those
model architecture hyperparameters, we recommend you to see their documents for more details.

Just like training :term:`tokenizer`, all arguments we used are just a mather of choice for training.  You can change
them to any values you want.

.. seealso::

   :py:mod:`lmp.model`
     All available :term:`language models`.

5. Evaluate Language Model
~~~~~~~~~~~~~~~~~~~~~~~~~~
Its time to check whether our :term:`language model` is successfully trained!

In this example we use Wiki-text-2 dataset to perform **validation** and **testing**.  But before that we should check
whether our model is **underfitting**.

.. code-block:: shell

   python -m lmp.script.evaluate_model_on_dataset wiki-text-2 \
     --batch_size 32 \
     --first_ckpt 0 \
     --exp_name my_model_exp \
     --ver train

We use **training** version of Wiki-Text-2 dataset (as specified in ``--ver train``) to check our performance.  The
script above will evaluate all :term:`checkpoints` we have saved starting from :term:`checkpoint` ``0`` all the way to
last :term:`checkpoint`.  We use :term:`perplexity` as our evaluation metric.  See :py:meth:`lmp.model.BaseModel.ppl`
for :term:`perplexity` details.

Again you can use browser to see your evaluation logs by the following script:

.. code-block::

   pipenv run tensorboard

After launch the command, you should open your **browser** and type http://localhost:6006/ to see your evaluation logs.
We will not write this script again later on.

If you didn't see the :term:`perplexity` goes down, this means your model is **underfitting**.  You should go back to
re-train your :term:`language model`.  Try using different batch size, number of epochs, and all sorts of
hyperparameters combination.

If you see the :term:`perplexity` goes down, that is good!  But how low should the :term:`perplexity` be?  To answer
that question, we recommed you to see the paper paired with the dataset (in some dataset they might not have papers to
reference).  But overall, lower than ``100`` might be a good indicator for a well-trained :term:`language model`.

We should now check whether our model is **overfitting**.

.. code-block:: shell

   python -m lmp.script.evaluate_model_on_dataset wiki-text-2 \
     --batch_size 32 \
     --first_ckpt 0 \
     --exp_name my_model_exp \
     --ver valid

We use **validation** version of Wiki-Text-2 dataset (as specified in ``--ver valid``) to check our performance.

If :term:`perplexity` on validation set does not do well, then we should go back to re-train our model, then validate
again, then re-train our model again, and so on.  The loop goes on and on until we reach a point where we get good
:term:`perplexity` on both training and validation dataset.  This means we might have a :term:`language model` which is
able to generalize on dataset we have never used to train (validation set in this case).  To further verify our
hypothesis, we should now use **test** version of Wiki-Text-2 dataset to check our performance.

.. code-block:: shell

   python -m lmp.script.evaluate_model_on_dataset wiki-text-2 \
     --batch_size 32 \
     --first_ckpt 0 \
     --exp_name my_model_exp \
     --ver test

6. Generate Text
~~~~~~~~~~~~~~~~
Finally we can use our well-trained :term:`language model` to generate text.  In this example we use
:py:mod:`lmp.script.generate_text` to generate text:

.. code-block:: shell

   python -m lmp.script.generate_text top-1 \
     --ckpt 5000 \
     --exp_name my_model_exp \
     --txt "We are"

We use ``top-1`` to specify we want to use :py:class:`lmp.infer.Top1Infer` as inference method to generate text.  We
use ``"We are"`` as condition text and generate text to complete the sentence or paragraph.

You can use different :term:`checkpoint` by changing the ``--ckpt 5000`` argument.  All available :term:`checkpoints`
is under :term:`experiment path` ``exp/my_model_exp``.  If :term:`checkpoint` does not exist then it will cause error.
Also if the models paired :term:`tokenizer` does not exist then it will cause error as well.

.. seealso::

   :py:mod:`lmp.infer`
     All available inference methods.

7. Record Experiment Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Now you have done the experiment, you can record them and compare experiments performed by others.  See
:doc:`Experiment Results <experiment/index>` for others' experiment and record yours!

Documents
---------
You can read documents on `this website`_ or use the following steps to build documents locally.  We use Sphinx_ to
build our documents.

.. _`this website`: https://language-model-playground.readthedocs.io/en/latest/index.html
.. _Sphinx: https://www.sphinx-doc.org/en/master/

1. Install documentation dependencies.

   .. code-block:: shell

      pipenv install --dev

2. Compile documents.

   .. code-block:: shell

      pipenv run doc

3. Open in the browser.

   .. code-block:: shell

      xdg-open doc/build/index.html


Testing
-------

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
