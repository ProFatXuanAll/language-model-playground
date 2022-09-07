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
  Let :math:`N = \set{0, 1, \dots, 99}` be the set of non-negative integers which are less than :math:`100`.
  Let :math:`a, b \in N`.

  +-----------+-------------------------------------------------------------------------+---------------+
  | Version   | Design Philosophy                                                       | Constraint    |
  +-----------+-------------------------------------------------------------------------+---------------+
  | ``train`` | Training set.                                                           | :math:`a < b` |
  +-----------+-------------------------------------------------------------------------+---------------+
  | ``valid`` | Check whether model learn commutative law on 2-digits integer addition. | :math:`a > b` |
  +-----------+-------------------------------------------------------------------------+---------------+
  | ``test``  | Check whether model learn to generalize 2-digits addition.              | :math:`a = b` |
  +-----------+-------------------------------------------------------------------------+---------------+

  Parameters
  ----------
  ver: Optional[str], default: None
    Version of the dataset.
    Set to ``None`` to use the default version ``self.__class__.df_ver``.

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

    train = []
    valid = []
    test = []
    for a in range(100):
      for b in range(100):
        if a < b:
          train.append(temp.format(str(a), str(b), str(a + b)))
        elif a > b:
          valid.append(temp.format(str(a), str(b), str(a + b)))
        else:
          test.append(temp.format(str(a), str(b), str(a + b)))

    if self.ver == 'train':
      self.spls = train
    elif self.ver == 'valid':
      self.spls = valid
    else:
      self.spls = test

    # Normalize dataset.
    self.spls = list(map(self.norm, self.spls))
