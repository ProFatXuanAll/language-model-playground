.. _my-reference-label:

QuickStart
=================

Setup environment
----------------------
1. Python version: 3.6+

2. CUDA version: 10.0+

Installation and Geting Started
--------------------------------------
1. Since the project contains big dataset, so we use git lfs to control versoin. Please insatll git lfs first, 
and execute

.. code-block:: guess


    $ git lfs install

2. Clone the project from github

.. code-block:: guess


    $ git clone https://github.com/ProFatXuanAll/language-model-playground.git

3. Move into the file

.. code-block:: guess


    $ cd language-model-playground

4. Install the dependency

.. code-block:: guess


    $ pipenv install

5. Start the virtual Environment

.. code-block:: guess


    $ pipenv shell

6. compile the document

.. code-block:: guess


    $ pipenv run doc

7. Open the document through browser

.. code-block:: guess


    $ xdg-open doc/build/index.html



Generating Document
------------------------------

1. Installation the Document dependency

.. code-block:: guess


    $ pipenv install --dev

2. Compile the Document

.. code-block:: guess


    $ pipenv run doc

3. Open in the browser

.. code-block:: guess


    $ xdg-open doc/build/index.html


Testing Language Model Playground
-------------------------------------
1. Installation the Document dependency

.. code-block:: guess


    $ pipenv install --dev

2. Execute the test

.. code-block:: guess


    $ isort .
    $ autopep8 -r -i -a -a -a lmp
    $ autopep8 -r -i -a -a -a test
    $ pipenv run test-coverage

Development Document
------------------------

1. Make sure your code conform Google python style guide.

2. Do type annotation for every function and method (You might need to see typing).

3. Write docstring for every class, function and method.

4. Run *$ pylint your_code.py* to automatically check your code whether conform to PEP 8.

5. Run *$ autopep8 -i -a -a your_code.py* to automatically fix your code and conform to PEP 8.

6. Run *$ mypy your_code.py* to check type annotaions.

7. Run *$ python -m unittest* to perform unit tests.

8. Write unit tests for your code and make them maintainable.