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

     apt install nvidia-driver-470

   .. note::

     You might need to use ``sudo`` to perform installation.

3. We use pipenv_ to install Python_ dependencies.
   You can install ``pipenv`` with

   .. code-block:: shell

     pip install pipenv

   .. warning::

     Do not use ``apt`` to intall pipenv_.

   .. note::

     You might want to set environment variable ``PIPENV_VENV_IN_PROJECT=1`` to make virtual environment folders always located in your Python_ projects.
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
   For example, you can take a look at chinese poem dataset by running :doc:`lmp.script.sample_dset </script/sample_dset>` :

   .. code-block:: shell

     python -m lmp.script.sample_dset chinese-poem

Training Language Model Pipline
-------------------------------
We now demonstrate a typical :term:`language model` training pipline.

.. note::

  Throughout this tutorial you might see the symbol ``\`` several times.
  ``\`` are used to format our CLI codes to avoid lenthy lines.
  All CLI codes can fit into one line.

1. Choose a Dataset
~~~~~~~~~~~~~~~~~~~
One have to choose a :term:`dataset` to :term:`train` a :term:`tokenizer` and a :term:`language model`.
In this example we use wiki-text-2 dataset :py:class:`~lmp.dset.WikiText2Dset` as demonstration.

.. seealso::

  :doc:`lmp.dset </dset/index>`
    All available datasets.
  :doc:`lmp.script.sample_dset </script/sample_dset>`
    Dataset sampling script.

2. Train a Tokenizer
~~~~~~~~~~~~~~~~~~~~
The following example use whitespace tokenizer :py:class:`~lmp.tknzr.WsTknzr` to train on :py:class:`~lmp.dset.WikiText2Dset` dataset since :term:`samples` in :py:class:`~lmp.dset.WikiText2Dset` are English and words are separated by whitespace.

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
  :doc:`lmp.script.train_tknzr </script/train_tknzr>`
    Tokenizer training script.

3. Evaluate Tokenizer
~~~~~~~~~~~~~~~~~~~~~
We can use :term:`pre-trained` :term:`tokenizer` to :term:`tokenize` arbitrary text.
In the following example we tokenize the sentence ``hello world`` into string list ``['hello', 'world']``:

.. code-block:: shell

  python -m lmp.script.tknz_txt \
    --exp_name my_tknzr_exp \
    --txt "hello world"

.. seealso::

  :doc:`lmp.script.tknz_txt </script/tknz_txt>`
    Text tokenization script.

4. Train a Language Model
~~~~~~~~~~~~~~~~~~~~~~~~~
Now we :term:`train` our :term:`language model` with the help of :term:`pre-trained` :term:`tokenizer`.
The following example train a LSTM (2000 version) based language model :py:class:`~lmp.model.LSTM2000`:

.. code-block:: shell

  python -m lmp.script.train_model LSTM-2000 \
    --batch_size 64 \
    --beta1 0.9 \
    --beta2 0.99 \
    --ckpt_step 1000 \
    --d_blk 64 \
    --d_emb 300 \
    --dset_name wiki-text-2 \
    --eps 1e-8 \
    --exp_name my_model_exp \
    --init_fb 1.0 \
    --init_ib -1.0 \
    --init_lower -0.1 \
    --init_ob -1.0 \
    --init_upper 0.1 \
    --label_smoothing 0.0 \
    --log_step 200 \
    --lr 1e-4 \
    --max_norm 1 \
    --max_seq_len 16 \
    --n_blk 8 \
    --n_lyr 1 \
    --p_emb 0.1 \
    --p_hid 0.1 \
    --stride 16 \
    --tknzr_exp_name my_tknzr_exp \
    --total_step 50000 \
    --ver train \
    --warmup_step 10000 \
    --weight_decay 1e-2

We log the training process with tensorboard_.
You can launch tensorboard and open browser with URL http://localhost:6006 to see the performance logs.
Use the following script to launch tensorboard:

.. code-block:: shell

  pipenv run tensorboard

.. seealso::

  :doc:`lmp.model </model/index>`
    All available language models.
  :doc:`lmp.script.train_model </script/train_model>`
    Language model training script.

5. Evaluate Language Model
~~~~~~~~~~~~~~~~~~~~~~~~~~
The following example use the validation set of wiki-text-2 dataset to evaluate :term:`language model`.

.. code-block:: shell

  python -m lmp.script.eval_dset_ppl wiki-text-2 \
    --batch_size 32 \
    --first_ckpt 0 \
    --exp_name my_model_exp \
    --ver valid

We log the evaluation process with tensorboard_.
You can launch tensorboard and open browser with URL http://localhost:6006 to see the performance logs.
Use the following script to launch tensorboard:

.. code-block:: shell

  pipenv run tensorboard

.. seealso::

  :doc:`lmp.script.eval_dset_ppl </script/eval_dset_ppl>`
    Dataset perplexity evaluation script.
  :doc:`lmp.script.eval_txt_ppl </script/eval_txt_ppl>`
    Text perplexity evaluation script.

6. Generate Continual Text
~~~~~~~~~~~~~~~~~~~~~~~~~~
We use :term:`pre-trained` :term:`language model` to generate continual text conditioned on given text segment ``I 'm on the highway to hell``.

.. code-block:: shell

  python -m lmp.script.gen_txt top-P \
    --ckpt -1 \
    --exp_name my_model_exp \
    --max_seq_len 128 \
    --p 0.9 \
    --txt "I 'm on the highway to hell"

.. seealso::

  :doc:`lmp.infer </infer/index>`
    All available inference methods.
  :doc:`lmp.script.gen_txt </script/gen_txt>`
    Continual text generation script.

7. Record Experiment Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Now you have finished an experiment.
You can record your results and compare results done by others.
See :doc:`Experiment Results <experiment/index>` for others' experiments and record yours!

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

.. footbibliography::

.. _PyTorch: https://pytorch.org/
.. _Python: https://www.python.org/
.. _CUDA: https://developer.nvidia.com/cuda-toolkit/
.. _pipenv: https://pipenv.pypa.io/en/latest/
.. _project: https://github.com/ProFatXuanAll/language-model-playground
.. _Sphinx: https://www.sphinx-doc.org/en/master/
.. _tensorboard: https://github.com/lanpa/tensorboardX
.. _`this website`: https://language-model-playground.readthedocs.io/en/latest/index.html
