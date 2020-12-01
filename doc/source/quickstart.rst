Quick Start
===========

We provide installation instructions only for Ubuntu ``18.04+`` (for now).

.. todo::

    Run test on Mac and Windows.

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

4. Since the project contains big dataset, we use ``git lfs`` to control
   dataset version.
   Please install ``git lfs`` first, and execute

   .. code-block:: shell

        git lfs install

.. _PyTorch: https://pytorch.org/

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

5. Now you can run any script under :py:mod:`lmp.script`!
   For example, you can take a look on chinese poem dataset by running
   :py:mod:`lmp.script.sample_from_dataset`

   .. code-block:: shell

        python -m lmp.script.sample_from_dataset --dset_name chinese-poem

Documents
---------

You can read documents on *this website* or use the following steps to build
documents locally.
We use Sphinx_ to build our documents.

.. _Sphinx: https://www.sphinx-doc.org/en/master/

.. todo::

    Publish documents on https://readthedocs.org/.

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
