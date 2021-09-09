Language Model Playground
==========================

Neural network based Language Model implemented with PyTorch.

.. _PyTorch: https://pytorch.org/

See documentation_ for more details.

.. _documentation: https://language-model-playground.readthedocs.io/en/
        latest/index.html

Environment Prerequest
----------------------
1. We only use python version ``3.8+``.
   You can install python with

   .. code-block:: shell

        apt-get install python3.8 python3.8-dev

2. We use PyTorch_ and thus use ``CUDA`` version: ``10.0+``.
   This only work if you have **Nvidia** GPUs.
   You can install ``CUDA`` library with

   .. code-block:: shell

        apt-get install nvidia-driver-450

3. We use ``pipenv`` to install dependencies.
   You can install ``pipenv`` with

   .. code-block:: shell

        pip install pipenv

Installation
------------

1. Clone the project from GitHub.

   .. code-block:: shell

        git clone https://github.com/ProFatXuanAll/language-model-playground.git

2. Change current directory to ``language-model-playground``.

   .. code-block:: shell

        cd language-model-playground

3. Install dependencies.
   We use ``pipenv`` to create virtual environment and install dependencies in
   virtual environment.

   .. code-block:: shell

        pipenv install

4. Start the virtual environment created by ``pipenv``.

   .. code-block:: shell

        pipenv shell

5. Now you can run any script under ``lmp.script``!
   For example, you can take a look on chinese poem dataset by running
   ``lmp.script.sample_from_dataset``

   .. code-block:: shell

        python -m lmp.script.sample_from_dataset chinese-poem

LICENSE
-------

Beerware license.
Anyone used this project must buy ProFatXuanAll_ a beer if you met him.

.. _ProFatXuanAll: https://github.com/ProFatXuanAll
