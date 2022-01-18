How To Document Guide
=====================

.. note::

   You need to install dev tools first by running

   .. code-block:: shell

      pipenv install --dev

Documentation Style Guide
-------------------------

We mostly follow numpydoc_ (which is extended from PEP257_) style guide.  Please RTFM.

.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard
.. _PEP257: https://www.python.org/dev/peps/pep-0257/

Autodoc
-------

We use `sphinx.ext.napoleon`_ (which use `sphinx.ext.autodoc`_) extension to generate document from docstring.  All
modules, classes, functions and scripts should be documented.  All following rule should be apply to document folder.

- Put ``lmp`` documents under ``doc/source/lmp``.
- Put ``test`` documents under ``doc/source/test``.

.. _`sphinx.ext.napoleon`: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
.. _`sphinx.ext.autodoc`: https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html

Build Document
--------------

To build document, we use sphinx_ to generate document in HTML format.  Run the following script::

  .. code-block:: shell

     pipenv run doc

.. _sphinx: https://www.sphinx-doc.org/en/master/#

Generate Document Coverage Report
---------------------------------

Run the following command in project root directory::

  .. code-block:: shell

     pipenv run doc-coverage

TODOs
~~~~~
We list some of the todos in the follow:

- Add ``pandas`` typeshed (mainly used by :py:class:`lmp.dset.ChPoemDset`) if it is stable.
- Remove ``# type: ignore`` on ``**kwargs`` if ``mypy`` can handle those checking (which may no happen forever).

