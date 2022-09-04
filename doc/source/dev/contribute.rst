Contributing to Language Model Playground
=========================================

We welcome all sorts of contributions to language model playground.
Please read the `Before You Commit`_ section first.
See `Need Helps`_ section for some known issues we currently need the most help.
For other issues such as **bugs** or **features request** see `our GitHub issues`_.

.. note::

  You need to install dev tools first by running

  .. code-block:: shell

    pipenv install --dev

Before You Commit
-----------------
#. Fork the project on GitHub.
#. Do **type annotation** for every functions, classes and methods.
   See :py:mod:`typing` and `PEP 484`_ for more information.

   Run the following script to check type annotaions:

   .. code-block:: shell

     mypy lmp test --ignore-missing-imports

   .. note::

     We aware that some types in :py:mod:`typing` is **deprecated** in favor of built-in syntax.
     For example, :py:mod:`typing.List` is deprecated in favor of built-in :py:class:`list`.
     However, those deprecated types will be in effect only after 2025 (See details in `PEP 585`_).
     Thus we will continue using :py:mod:`typing` instead of built-in syntax.
     Also we are using Python_ version ``3.8`` which does not support built-in syntax.
     We will switch to built-in syntax once PyTorch_ support Python_ version ``3.9``.

#. Write docstring for every modules, classes, functions and methods.
   See :doc:`how_to_doc` for detailed docstring guide line.
#. Run the following script to lint your code to conform `PEP 8`_.

   .. code-block:: shell

     pipenv run ly
     pipenv run li
     pipenv run lf

   Fix any error/warning messages showed on CLI.
   If you find some rule is not possible to be fix, please **open issue**.

#. Run tests and get test coverage report.
   Make sure your codes do not break existing code.
   See :doc:`how_to_test` for detailed testing guide line.
#. Write tests for your codes and make them maintainable.
   See :doc:`how_to_test` for detailed testing guide line.

Need Helps
~~~~~~~~~~
The following list of items are the helps we needed.

- Translate documents into traditional Chinese.
- Require installation script.
  Currently python is undergone throught some major change on their package management system.
  But we think its better to stick to ``setup.py`` solution and change to better solution after the change of package management system.

.. footbibliography::

.. _`our GitHub issues`: https://github.com/ProFatXuanAll/language-model-playground/issues
.. _`PEP 8`: https://www.python.org/dev/peps/pep-0008/
.. _`PEP 484`: https://www.python.org/dev/peps/pep-0484/
.. _`PEP 585`: https://www.python.org/dev/peps/pep-0585/
.. _PyTorch: https://pytorch.org/
.. _Python: https://www.python.org/
