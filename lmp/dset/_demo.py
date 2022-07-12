"""Demo dataset."""

from typing import ClassVar, List, Optional

from lmp.dset._base import BaseDset


class DemoDset(BaseDset):
  r"""Demo dataset.

  This dataset is consist of 2-digits addition literatures.
  All literatures have the following format:

    If you add :math:`a` to :math:`b` you get :math:`a + b` .

  where :math:`a, b` are integers within :math:`0` to :math:`99` (inclusive).

  Here we describe the dataset in detail.
  Let :math:`E = \set{0, 2, 4, \dots, 98}` be the set of non-negative, less than 100 even numbers, and let
  :math:`E = \set{1, 3, 5, \dots, 99}` be the set of positive, less than 100 odd numbers.

  +-----------+---------------------------------------+--------------------+--------------------+
  | Version   | Design Philosophy                     | Range of :math:`a` | Range of :math:`b` |
  +-----------+---------------------------------------+--------------------+--------------------+
  | ``train`` | Train the model.                      | :math:`a \in E`    | :math:`b \in O`    |
  +-----------+---------------------------------------+--------------------+--------------------+
  | ``valid`` | Check whether model learn commutative | :math:`a \in O`    | :math:`b \in E`    |
  |           | law of 2-digits integer addition.     |                    |                    |
  +-----------+---------------------------------------+--------------------+--------------------+
  | ``test``  | Check whether model learn to          | :math:`a = b` and                       |
  |           | generalize 2-digits addition.         | :math:`a \in E \cup O`                  |
  +-----------+---------------------------------------+-----------------------------------------+

  Parameters
  ----------
  ver: Optional[str], default: None
    Version of the dataset.
    Set ``ver = ''`` to use default version.

  Attributes
  ----------
  df_ver: typing.ClassVar[str]
    Default version is ``'train'``.
  dset_name: typing.ClassVar[str]
    CLI name of demo dataset is ``demo``.
  spls: list[str]
    All samples in the dataset.
  ver: str
    Version of the dataset.
  vers: typing.ClassVar[list[str]]
    Supported versions including ``'train'``, ``'test'`` and ``'valid'``.

  See Also
  --------
  :doc:`lmp.dset </dset/index>`
    All available datasets.

  Examples
  --------
  >>> from lmp.dset import DemoDset
  >>> dset = DemoDset(ver='train')
  >>> dset[0]
  'If you add 0 to 1 you get 1 .'
  """

  df_ver: ClassVar[str] = 'train'
  dset_name: ClassVar[str] = 'demo'
  vers: ClassVar[List[str]] = ['test', 'train', 'valid']

  def __init__(self, *, ver: Optional[str] = None):
    super().__init__(ver=ver)

    # Demo text template.
    temp = 'If you add {} to {} you get {} .'

    # Number ranges in demo text.
    even = list(range(0, 100, 2))
    odd = list(range(1, 100, 2))

    if self.ver == 'train':
      for num_1 in even:
        for num_2 in odd:
          self.spls.append(temp.format(str(num_1), str(num_2), str(num_1 + num_2)))
    elif self.ver == 'valid':
      # Validation set is used to test commutitive law.
      for num_1 in odd:
        for num_2 in even:
          self.spls.append(temp.format(str(num_1), str(num_2), str(num_1 + num_2)))
    else:
      # Test set is used to test multiplication
      for num in even + odd:
        self.spls.append(temp.format(str(num), str(num), str(2 * num)))

    # Normalize dataset.
    self.spls = list(map(self.norm, self.spls))

    # Sort dataset by length in ascending order.
    self.spls.sort(key=len)
