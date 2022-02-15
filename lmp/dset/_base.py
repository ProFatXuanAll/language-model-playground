"""Dataset base class."""

import os
import re
import unicodedata
from typing import ClassVar, Iterator, List, Optional

import requests
import torch.utils.data

import lmp.util.path
import lmp.util.validate


class BaseDset(torch.utils.data.Dataset):
  """Dataset base class.

  Most datasets need to be downloaded from the web, some of them can be generated locally.  If requested datasets are
  not on your local machine, then they will be downloaded / generated automatically.  Once dataset files exist they
  will not be downloaded / generated again.

  Parameters
  ----------
  ver: Optional[str], default: None
    Version of the dataset.  Set to ``None`` to use default version ``self.__class__.df_ver``.

  Attributes
  ----------
  df_ver: typing.ClassVar[str]
    Default version of the dataset.
  dset_name: typing.ClassVar[str]
    CLI Display name of the dataset.  Only used to parse CLI arguments.
  spls: list[str]
    All samples in the dataset.
  ver: str
    Version of the dataset.
  vers: typing.ClassVar[list[str]]
    List of supported datasets' versions.

  See Also
  --------
  :doc:`lmp.dset </dset/index>`
    All available datasets.
  """

  df_ver: ClassVar[str] = ''
  dset_name: ClassVar[str] = 'base'
  vers: ClassVar[List[str]] = []

  def __init__(self, *, ver: Optional[str] = None):
    super().__init__()
    # Use default version of the dataset.
    if ver is None:
      ver = self.__class__.df_ver

    # `ver` validation.
    lmp.util.validate.raise_if_not_instance(val=ver, val_name='ver', val_type=str)
    lmp.util.validate.raise_if_not_in(val=ver, val_name='ver', val_range=self.__class__.vers)
    self.ver = ver

    self.spls: List[str] = []

  def __iter__(self) -> Iterator[str]:
    """Iterate through each sample in the dataset.

    Yields
    ------
    str
      One sample in ``self.spls``, ordered by sample indices.
    """
    for spl in self.spls:
      yield spl

  def __len__(self) -> int:
    """Get dataset size.

    Returns
    -------
    int
      Number of samples in the dataset.
    """
    return len(self.spls)

  def __getitem__(self, idx: int) -> str:
    """Sample text using index.

    Parameters
    ----------
    idx: int
      Sample index.

    Returns
    -------
    str
      The sample which index equals to ``idx``.
    """
    # `idx` validation.
    lmp.util.validate.raise_if_not_instance(val=idx, val_name='idx', val_type=int)
    return self.spls[idx]

  @staticmethod
  def download_file(mode: str, download_path: str, url: str) -> None:
    """Download file from ``url``.

    Arguments
    ---------
    mode: str
      Can only be ``'binary'`` or ``'text'``.
    download_path: str
      File path of the downloaded file.
    url: str
      URL of the file to be downloaded.

    Returns
    -------
    None
    """
    # `mode` type guard.
    lmp.util.validate.raise_if_not_instance(val=mode, val_name='mode', val_type=str)
    lmp.util.validate.raise_if_not_in(val=mode, val_name='mode', val_range=['text', 'binary'])

    # `download_path` type guard.
    lmp.util.validate.raise_if_not_instance(val=download_path, val_name='download_path', val_type=str)
    lmp.util.validate.raise_if_empty_str(val=download_path, val_name='download_path')
    lmp.util.validate.raise_if_is_directory(path=download_path)

    # `url` type guard.
    lmp.util.validate.raise_if_not_instance(val=url, val_name='url', val_type=str)
    lmp.util.validate.raise_if_empty_str(val=url, val_name='url')

    # Create folder if not exists.
    download_dir = os.path.abspath(os.path.join(download_path, os.pardir))
    if not os.path.exists(download_dir):
      os.makedirs(download_dir)

    # Download and output file.
    if mode == 'binary':
      with requests.get(url=url) as res, open(download_path, 'wb') as binary_file:
        binary_file.write(res.content)
    else:
      with requests.get(url=url) as res, open(download_path, 'w', encoding='utf-8') as text_file:
        text_file.write(res.text)

  @staticmethod
  def norm(txt: str) -> str:
    """Text normalization.

    Text will be NFKC normalized.  Whitespaces are collapsed and strip from both ends.

    Parameters
    ----------
    txt: str
      Text to be normalized.

    Returns
    -------
    str
      Normalized text.

    See Also
    --------
    unicodedata.normalize
      Python built-in unicode normalization.

    Examples
    --------
    >>> from lmp.dset import BaseDset
    >>> BaseDset.norm('１２３４５６７８９')
    '123456789'
    """
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', txt).strip())
