How To Test Guide
=================

.. note::

  You need to install dev tools first by running

  .. code-block:: shell

    pipenv install --dev

Test Style Guide
----------------
We use pytest_ testing framework to test our code.
We use coverage_ to provide test coverage report.
Please RTFM.

Run Test
--------
Run the following command in project root directory:

.. code-block:: shell

  pipenv run test

.. note::
  All dataset will be downloaded to local disk and will not be removed after test.
  This speed up testing and is the desired behavior.

Generate Test Coverage Report
-----------------------------
Run the following command in project root directory:

.. code-block:: shell

  pipenv run test-coverage

.. todo::

  Add more test writing guide line.

.. footbibliography::

.. _pytest: https://docs.pytest.org/en/reorganize-docs/contents.html
.. _coverage: https://coverage.readthedocs.io/en/coverage-5.3/index.html
