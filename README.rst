Language Model Playground
==========================

Neural network based Language Model implemented with PyTorch.  See documentation_ for more details.

Environment Prerequest
----------------------
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

LICENSE
-------
Beerware license.
Anyone used this project must buy ProFatXuanAll_ a beer if you met him.

.. _CUDA: https://developer.nvidia.com/cuda-toolkit/
.. _ProFatXuanAll: https://github.com/ProFatXuanAll
.. _PyTorch: https://pytorch.org/
.. _Python: https://www.python.org/
.. _documentation: https://language-model-playground.readthedocs.io/en/latest/index.html
.. _pipenv: https://pipenv.pypa.io/en/latest/
.. _project: https://github.com/ProFatXuanAll/language-model-playground.git
