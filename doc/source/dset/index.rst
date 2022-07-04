Datasets
========

We have collected some datasets and provide utilities for accessing dataset samples.
To access dataset samples, one first need to create a dataset instance, then use index to access dataset samples.

.. code-block:: python

   import lmp.dset

   # Create dataset instance.
   dataset = lmp.dset.DemoDset(ver='train')

   # Access samples by index.
   sample_0 = dataset[0]
   sample_1 = dataset[1]

You can use :py:func:`len` to get the number of samples in dataset.
And you can enumerate samples by treating dataset as iterator.

.. code-block:: python

   import lmp.dset

   # Use ``len`` to get dataset size.
   dataset = lmp.dset.DemoDset(ver='train')
   dataset_size = len(dataset)

   # Use dataset as iterator.
   for sample in dataset:
      print(sample)

Dataset may have many versions.
You can see the supported versions by the :py:attr:`lmp.dset.BaseDset.vers` class attribute.
Default version is defined as :py:attr:`lmp.dset.BaseDset.df_ver`.

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

   # Get default version.
   assert 'train' == lmp.dset.DemoDset.df_ver

All available datasets are listed below:

.. toctree::
   :glob:
   :maxdepth: 1

   *
