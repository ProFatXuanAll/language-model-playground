Datasets
========

Overview
--------
When :term:`training` a :term:`model`, one must first collect a :term:`dataset` and preprocess it so that
text :term:`samples` have certain structure / format.
In this project we have collect some datasets and provide utilities so that one can train :term:`language models` on
them.

.. seealso::

  :doc:`lmp.script.sample_dset </script/sample_dset>`
    Dataset sampling script.

Import dataset module
---------------------
All dataset class are collectively gathered as a module :py:mod:`lmp.dset`.
One can import dataset module as usual Python_ module:

.. code-block :: python

  import lmp.dset

Create dataset instances
------------------------
After importing dataset module :py:mod:`lmp.dset`, one can create dataset instance through the class attributes of
:py:mod:`lmp.dset`.
For example, one can create demo dataset :py:class:`lmp.dset.DemoDset` as follow:

.. code-block:: python

  import lmp.dset

  # Create demo dataset instance.
  dataset = lmp.dset.DemoDset()

A dataset can have many different version.
You can see all supported versions by the dataset class attribute ``vers``.
For example, all the supported versions of :py:class:`lmp.dset.DemoDset` are ``test``, ``train`` and ``valid``.

.. code-block:: python

  import lmp.dset

  # Supported versions of `lmp.dset.DemoDset`.
  assert 'test' in lmp.dset.DemoDset.vers
  assert 'train' in lmp.dset.DemoDset.vers
  assert 'valid' in lmp.dset.DemoDset.vers

  # Construct different versions.
  test_dataset = lmp.dset.DemoDset(ver='test')
  train_dataset = lmp.dset.DemoDset(ver='train')
  valid_dataset = lmp.dset.DemoDset(ver='valid')

If ``ver`` parameter is not given to a dataset class, then the default version of a dataset class is used.
The default version of a dataset class is defined as the class attribute ``df_ver``.
For example, the default version of :py:class:`DemoDset` is ``train``:

.. code-block:: python

  import lmp.dset

  # Get default version.
  assert 'train' == lmp.dset.DemoDset.df_ver

  # The following constructions are all the same.
  train_dataset = lmp.dset.DemoDset()
  train_dataset = lmp.dset.DemoDset(ver=None)
  train_dataset = lmp.dset.DemoDset(ver='train')
  train_dataset = lmp.dset.DemoDset(ver=lmp.dset.DemoDset.df_ver)

Sample dataset
--------------
To access dataset :term:`samples`, one first need to create a dataset instance, then use integer indices to access
dataset samples.
For example, we can access the 0th and 1st samples in the training set of :py:class:`lmp.dset.DemoDset` as follow:

.. code-block:: python

  import lmp.dset

  # Create dataset instance.
  dataset = lmp.dset.DemoDset(ver='train')

  # Access samples by indices.
  sample_0 = dataset[0]
  sample_1 = dataset[1]

One can use :py:func:`len` to get the total number of samples in a dataset.
For example, we can access each sample in :py:class:`lmp.dset.DemoDset` as follow:

.. code-block:: python

  import lmp.dset

  # Use ``len`` to get dataset size.
  dataset = lmp.dset.DemoDset(ver='train')
  dataset_size = len(dataset)

  # Access each sample in the dataset.
  for index in range(dataset_size):
    print(dataset[index])

One can enumerate samples by treating a dataset instance as an iterator.
For example, we can iterate through each sample in :py:class:`lmp.dset.DemoDset` as follow:

.. code-block:: python

  import lmp.dset

  # Use dataset as iterator.
  for sample in lmp.dset.DemoDset(ver='train'):
    print(sample)

.. seealso::

  :doc:`lmp.script.sample_dset </script/sample_dset>`
    Dataset sampling script.

Download dataset
----------------
We have provided downloading utilities so that if a dataset is not on your local machine it will be downloaded
automatically.
All downloaded files will be put under ``data`` directory.
For example, to download the training set of :py:class:`lmp.dset.Wiki-text-2`, all you need to do is as follow:

.. code-block:: python

  import lmp.dset

  # Automatically download dataset if dataset does not exist on local machine.
  dataset = lmp.dset.WikiTextDset(ver='train')

All available datasets
----------------------
.. toctree::
  :glob:
  :maxdepth: 1

  *

.. _Python: https://www.python.org/
