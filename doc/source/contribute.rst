Contributing to Language Model Playground
=========================================

We welcome all sorts of contributions to language model playground.
Please read the `Before You Commit`_ section **before** you do anything.
See `Need Helps`_ section for some know issues we currently need help the most.
For other issues such as **bugs** or **features request** see
`our GitHub issues`_.

.. _`our GitHub issues`: https://github.com/ProFatXuanAll/
    language-model-playground/issues

.. note::

    You need to install dev tools first by running

    .. code-block:: sh

        pipenv install --dev

Before You Commit
-----------------

#. Fork the project on GitHub.
#. Do **type annotation** for every functions, classes and methods.
   See :py:mod:`typing` and `PEP 484`_ for more information.

   Run the following script to check type annotaions:

   .. code-block:: sh

        mypy lmp test --ignore-missing-imports

   .. note::

        We aware that some types in :py:mod:`typing` is **deprecated** in
        favor of built-in syntax.
        For example, :py:mod:`typing.List` is deprecated in favor of built-in
        :py:class:`list`.
        However, those deprecated types will be in effect only after 2025
        (See details in `PEP 585`_).
        Thus we will continue using :py:mod:`typing` instead built-in syntax
        since the :py:mod:`typing` namespace provide more hint for type
        annotation purpose.

   .. note::

        You will see lots of error after running ``mypy`` script.
        This is expected since ``mypy`` can only do static type checking.
        False positive errors and warnings should be ignored.

#. Write docstring for every class, function and method.
   See :doc:`how_to_doc` for detailed docstring guide line.

#. Run the following script to lint your code to conform `PEP 8`_.

   .. code-block:: sh

        autopep8 -i -r -a -a -a lmp test
        flake8 lmp test

   Fix any error/warning messages showed on CLI.
   If you find some rule is not possible to be fix, please **open issue**.

#. Run the following script to sort imports alphabetically, 
   and automatically separated into sections and by type.

   .. code-block:: sh

        isort test

        
#. Run tests and get test coverage report.
   Make sure your code do not break existing code.
   See :doc:`how_to_test` for detailed testing guide line.

#. Write tests for your code and make them maintainable.
   See :doc:`how_to_test` for detailed testing guide line.

.. _`PEP 8`: https://www.python.org/dev/peps/pep-0008/
.. _`PEP 484`: https://www.python.org/dev/peps/pep-0484/
.. _`PEP 585`: https://www.python.org/dev/peps/pep-0585/

Need Helps
~~~~~~~~~~
The following list of items are the helps we needed.

- Unittest on functions.
- Unittest on classes.
- Unittest on scripts.
- Documentation translation to traditional Chinese.
  We also need to separate English documents from traditional Chinese.
- Require installation script.
  Currently python is undergone throught some major change on their package
  management system.
  But we think its better to stick to ``setup.py`` solution and change to
  better solution after the change of package management system.
