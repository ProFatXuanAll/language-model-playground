Quick start
=================

Setup environment
----------------------
1. Python version: ``3.8``

2. CUDA version: 10.0+

Installation and Geting Started
--------------------------------------
1. Since the project contains big dataset, we use ``git lfs`` to control versoin. Please install ``git lfs`` first, 
and execute

.. code-block:: shell

    git lfs install

2. Clone the project from github

.. code-block:: shell

    git clone https://github.com/ProFatXuanAll/language-model-playground.git

3. Change current directory

.. code-block:: shell

    cd language-model-playground

4. Install the dependency

.. code-block:: shell

    pipenv install --dev

5. Start the virtual Environment

.. code-block:: shell

    pipenv shell

6. compile the document

.. code-block:: shell

    pipenv run doc

7. Open the document through browser

.. code-block:: shell

    xdg-open doc/build/index.html



Generating Document
------------------------------

1. Installation the Document dependency

.. code-block:: shell

    pipenv install --dev

2. Compile the Document

.. code-block:: shell

    pipenv run doc

3. Open in the browser

.. code-block:: shell

    xdg-open doc/build/index.html


Testing Language Model Playground
-------------------------------------
1. Installation the Document dependency

.. code-block:: shell

    pipenv install --dev

2. Execute the test

.. code-block:: shell

    isort .
    autopep8 -r -i -a -a -a lmp
    autopep8 -r -i -a -a -a test
    pipenv run test
    pipenv run test-coverage

Development Document
------------------------

1. Make sure your code conform `numpydoc docstring guide. <https://numpydoc.readthedocs.io/en/latest/format.html>`_ 

2. Do type annotation for every function and method (You might need to see `typing <https://docs.python.org/3/library/typing.html>`_).

3. Write docstring for every class, function and method.

4. Run ``pylint your_code.py`` to automatically check your code whether conform to `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_.

5. Run ``autopep8 -i -a -a your_code.py`` to automatically fix your code and conform to `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_.

6. Run ``mypy your_code.py`` to check type annotaions.

7. Run ``python -m unittest`` to perform unit tests.

8. Write unit tests for your code and make them maintainable.