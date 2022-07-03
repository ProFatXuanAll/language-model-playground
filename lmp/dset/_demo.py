"""Demo dataset."""

from typing import ClassVar, List, Optional

from lmp.dset._base import BaseDset


class DemoDset(BaseDset):
  r"""Demo dataset.

  This dataset is consist of literatures of 2-digits additions.
  The literatures are in the following format:

    If you add :math:`a` to :math:`b` you get :math:`a + b` .

  where :math:`a, b` are integers within :math:`0` to :math:`99` (inclusive).

  Here we describe the dataset in detail.
  Let :math:`E_1 = \set{0, 2, \dots, 48}`, :math:`E_2 = \set{50, 52, \dots, 98}`, :math:`O_1 = \set{1, 3, \dots, 49}`
  and :math:`O_2 = \set{51, 53, \dots, 99}`.

  +-----------+---------------------------------------+----------------------+-----------------------+
  | Version   | Design Philosophy                     | Range of :math:`a`   | Range of :math:`b`    |
  +-----------+---------------------------------------+----------------------+-----------------------+
  | ``train`` | Train the model.                      | :math:`E_1 \cup O_2` | :math:`E_2 \cup O_1`  |
  +-----------+---------------------------------------+----------------------+-----------------------+
  | ``valid`` | Check whether model learn commutative | :math:`E_2 \cup O_1` | :math:`E_1 \cup O_2`  |
  |           | law of 2-digits integer addition.     |                      |                       |
  +-----------+---------------------------------------+----------------------+-----------------------+
  | ``test``  | Check whether model learn to          | :math:`a = b` and                            |
  |           | generalize 2-digits addition.         | :math:`a \in E_1 \cup E_2 \cup O_1 \cup O_2` |
  +-----------+---------------------------------------+----------------------------------------------+

  Parameters
  ----------
  ver: Optional[str], default: None
    Version of the dataset.   Set ``ver = ''`` to use default version.

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
  lmp.dset.BaseDset
    Dataset utilities.

  Examples
  --------
  >>> from lmp.dset import DemoDset
  >>> dset = DemoDset(ver='train')
  >>> dset[0]
  'If you add 0 to 50 you get 50 .'
  """

  df_ver: ClassVar[str] = 'train'
  dset_name: ClassVar[str] = 'demo'
  vers: ClassVar[List[str]] = ['test', 'train', 'valid']

  def __init__(self, *, ver: Optional[str] = None):
    super().__init__(ver=ver)

    # Demo text template.
    temp = 'If you add {} to {} you get {} .'

    # Number ranges in demo text.
    even_0_48 = list(range(0, 50, 2))
    even_50_98 = list(range(50, 100, 2))
    odd_1_49 = list(range(1, 50, 2))
    odd_51_99 = list(range(51, 100, 2))

    if self.ver == 'train':
      for num_1 in even_0_48 + odd_51_99:
        for num_2 in even_50_98 + odd_1_49:
          self.spls.append(temp.format(str(num_1), str(num_2), str(num_1 + num_2)))
    elif self.ver == 'valid':
      # Validation set is used to test commutitive law.
      for num_1 in even_0_48 + odd_51_99:
        for num_2 in even_50_98 + odd_1_49:
          self.spls.append(temp.format(str(num_2), str(num_1), str(num_1 + num_2)))
    else:
      # Test set is used to test multiplication
      for num in even_0_48 + even_50_98 + odd_1_49 + odd_51_99:
        self.spls.append(temp.format(str(num), str(num), str(2 * num)))

    # Normalize dataset.
    self.spls = list(map(self.norm, self.spls))
