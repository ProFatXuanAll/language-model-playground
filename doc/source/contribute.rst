Contributing to Language Model Playground
=========================================

We welcome any contribution to language model playground.
We are currently doing project restructure.
It might be better to contribute after we finish restructuring.

.. todo::

   Remove restructure hint and add more **need help** section.

Before You Commit
-----------------

#. Do **type annotation** for every functions, classes and methods.
   See :py:mod:`typing` and `PEP 484`_ for more information.

   .. note::

        We aware that some types in :py:mod:`typing` is **deprecated** in
        favor of built-in syntax.
        For example, :py:mod:`typing.List` is deprecated in favor of `list`.
        However, those deprecated types will be in effect only after 2025
        (See details in `PEP 585`_).
        Thus we will continue using :py:mod:`typing` instead built-in syntax
        since the :py:mod:`typing` namespace provide more hint for type
        annotation purpose.

#. Write docstring for every class, function and method.
   See :doc:`how_to_doc` for detailed docstring guide line.

#. Run ``autopep8 -i -a -a -a lmp test`` to automatically lint your code to
   conform `PEP 8`_.
   ``autopep8`` might not be able to catch all syntax problems, for further
   linting see steps below.

#. Run ``flake8 lmp test`` to lint your code to conform `PEP 8`_.

#. Run ``mypy lmp test`` to check type annotaions.

#. Run tests and get test coverage report.
   Make sure your code do not break existing code.
   See :doc:`how_to_test` for detailed testing guide line.

#. Write tests for your code and make them maintainable.
   See :doc:`how_to_test` for detailed testing guide line.


.. _`PEP 8`: https://www.python.org/dev/peps/pep-0008/
.. _`PEP 484`: https://www.python.org/dev/peps/pep-0484/
.. _`PEP 585`: https://www.python.org/dev/peps/pep-0585/