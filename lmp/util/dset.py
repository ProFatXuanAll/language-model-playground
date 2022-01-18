"""Dataset utilities."""

from typing import Dict, Optional

import lmp.util.validate
from lmp.dset import DSET_OPTS, BaseDset


def load(dset_name: str, ver: str, **kwargs: Optional[Dict]) -> BaseDset:
  """Load dataset.

  Parameters
  ----------
  dset_name: str
    Name of the dataset to be loaded.
  ver: str
    Version of the dataset to be loaded.

  Returns
  -------
  lmp.dset.BaseDset
    Loaded dataset instance.

  See Also
  --------
  lmp.dset
    All available datasets.

  Examples
  --------
  >>> from lmp.dset import WikiText2Dset
  >>> import lmp.util.dset
  >>> dset = lmp.util.dset.load(dset_name='wiki-text-2', ver='train')
  >>> isinstance(dset, WikiText2Dset)
  True
  >>> dset.ver == 'train'
  True
  """
  # `dset_name` validation.
  lmp.util.validate.raise_if_not_instance(val=dset_name, val_name='dset_name', val_type=str)
  lmp.util.validate.raise_if_not_in(val=dset_name, val_name='dset_name', val_range=list(DSET_OPTS.keys()))

  # `ver` will be validated in `BaseDset.__init__`.
  return DSET_OPTS[dset_name](ver=ver)
