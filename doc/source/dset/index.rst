Datasets
========

Overview
--------
When :term:`training` a :term:`model`, one must first collect a :term:`dataset` and preprocess it so that text :term:`samples` have certain structure / format.
In this project we have collect some datasets and provide utilities so that one can train :term:`language models` on them.

.. seealso::

  :doc:`lmp.script.sample_dset </script/sample_dset>`
    Dataset sampling script.

Import dataset module
---------------------
All :term:`dataset` classes are collectively gathered under the module :py:mod:`lmp.dset`.
One can import dataset module as usual Python_ module:

.. code-block:: python

  import lmp.dset

Create dataset instances
------------------------
After importing :py:mod:`lmp.dset`, one can create :term:`dataset` instance through the class attributes of :py:mod:`lmp.dset`.
For example, one can create demo dataset :py:class:`lmp.dset.DemoDset` and wiki-text-2 dataset :py:class:`lmp.dset.WikiText2Dset` as follow:

.. code-block:: python

  import lmp.dset

  # Create demo dataset instance.
  demo_dataset = lmp.dset.DemoDset()

  # Create wiki-text-2 dataset instance.
  wiki_dataset = lmp.dset.WikiText2Dset()

A dataset can have many versions.
One can access the class attribute ``vers`` of a dataset class to get all supported versions.
For example, all supported versions of :py:class:`lmp.dset.DemoDset` are ``test``, ``train`` and ``valid``:

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

If parameter ``ver`` is not passed to dataset class's constructor, the default version of a dataset class is used.
The default version of a dataset class is defined as the class attribute ``df_ver``.
For example, the default version of :py:class:`DemoDset` is ``train``:

.. code-block:: python

  import lmp.dset

  # Get default version.
  assert 'train' == lmp.dset.DemoDset.df_ver

  # All following constructions are the same.
  train_dataset = lmp.dset.DemoDset()
  train_dataset = lmp.dset.DemoDset(ver=None)
  train_dataset = lmp.dset.DemoDset(ver='train')
  train_dataset = lmp.dset.DemoDset(ver=lmp.dset.DemoDset.df_ver)

Sample from dataset
-------------------
One can access dataset :term:`samples` through dataset instances.
The only way to access specific sample is using indices.
For example, we can access the 0th and the 1st samples in the training set of :py:class:`lmp.dset.DemoDset` as follow:

.. code-block:: python

  import lmp.dset

  # Create dataset instance.
  dataset = lmp.dset.DemoDset(ver='train')

  # Access samples by indices.
  sample_0 = dataset[0]
  sample_1 = dataset[1]

One can use :py:func:`len` to get the total number of samples in a dataset.
For example, we can enumerate each sample in :py:class:`lmp.dset.DemoDset` as follow:

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
We provide downloading utilities so that datasets are downloaded automatically if they are not on your local machine.
All downloaded files will be put under ``project_root/data`` directory.
For example, to download the training set of :py:class:`lmp.dset.WikiText2Dset`, all you need to do is as follow:

.. code-block:: python

  import lmp.dset

  # Automatically download dataset if dataset is not on local machine.
  dataset = lmp.dset.WikiTextDset(ver='train')

All available datasets
----------------------
.. toctree::
  :glob:
  :maxdepth: 1

  *

.. footbibliography::

.. _Python: https://www.python.org/
