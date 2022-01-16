"""Wiki-Text-2 dataset."""

import io
import os
import re
import zipfile
from typing import ClassVar, List, Optional

import lmp.util.path
from lmp.dset._base import BaseDset


class WikiText2Dset(BaseDset):
  """Wiki-Text-2 dataset.

  [Wiki-Text-2]_ is an English dataset which is part of the WikiText Long Term Dependency Language Modeling Dataset.
  See WikiText_ for more details on dataset.

  .. _WikiText: https://www.salesforce.com/products/einstein/ai-research/
    the-wikitext-dependency-language-modeling-dataset/

  Here are supported versions and number of tokens informations.

  +-----------+--------------------------+--------------------------+
  | version   | maximum number of tokens | minimum number of tokens |
  +===========+==========================+==========================+
  | ``train`` | 2602                     | 2                        |
  +-----------+--------------------------+--------------------------+
  | ``test``  | 2959                     | 1                        |
  +-----------+--------------------------+--------------------------+
  | ``valid`` | 1775                     | 1                        |
  +-----------+--------------------------+--------------------------+

  Parameters
  ----------
  ver: str, default: None
    Version of the dataset.   Set ``ver = ''`` to use default version.

  Attributes
  ----------
  df_ver: ClassVar[str]
    Default version is ``'train'``.
  dset_name: ClassVar[str]
    Wiki-Text-2 dataset's name is ``wiki-text-2``.
  spls: List[str]
    All samples in the dataset.
  ver: str
    Version of the dataset.
  vers: ClassVar[List[str]]
    Supported versions including ``'train'``, ``'test'`` and ``'valid'``.

  See Also
  --------
  lmp.dset
    All available datasets.
  lmp.dset.BaseDset
    Dataset utilities.

  References
  ----------
  .. [Wiki-Text-2] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2016. Pointer Sentinel Mixture
    Models

  Examples
  --------
  >>> from lmp.dset import WikiText2Dset
  >>> dset = WikiText2Dset(ver='test')
  >>> dset[0][:31]
  'Robert [unk] is an English film'
  """

  df_ver: ClassVar[str] = 'train'
  dset_name: ClassVar[str] = 'wiki-text-2'
  vers: ClassVar[List[str]] = ['test', 'train', 'valid']

  def __init__(self, *, ver: Optional[str] = None):
    super().__init__(ver=ver)

    # Make sure dataset files exist.
    self.download_dataset()

    # Read dataset from the specified version.
    with open(os.path.join(lmp.util.path.DATA_PATH, f'wiki.{self.ver}.tokens'), 'r') as text_file:
      lines = [line.strip() for line in text_file.readlines()]

    spl = ''
    for line in lines:
      # Discard all empty lines and (sub)section titles.
      if not line or line.startswith('='):
        # Perform text normalization and replace unknown token `<unk>` with `[unk]`.
        spl = re.sub(r'<unk>', '[unk]', self.norm(spl))
        # Avoid including empty sample.
        if spl:
          self.spls.append(spl)
        # Reset to record next sample.
        spl = ''
        continue

      # Nearby paragraphs (separated by one newline) are concatenated to form one sample.
      spl = f'{spl} {line}'

  @classmethod
  def download_dataset(cls) -> None:
    """Download Wiki-text-2 dataset.

    Download zip file from https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip and extract raw
    files from zip file.  Raw files are named as ``'wiki.ver.tokens'``, where ``ver`` is the version of the dataset.
    After extracting raw files the downloaded zip file will be deleted.

    Returns
    -------
    None
    """
    # Download zip file path.
    zip_file_path = os.path.join(lmp.util.path.DATA_PATH, 'wiki-text-2.zip')
    # Original source is no longer available.
    url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'

    # Avoid duplicated download by checking whether all raw files exists.
    already_downloaded = True
    for ver in cls.vers:
      raw_file_path = os.path.join(lmp.util.path.DATA_PATH, f'wiki.{ver}.tokens')
      if not os.path.exists(raw_file_path):
        already_downloaded = False

    if already_downloaded:
      return

    # Download dataset.
    BaseDset.download_file(mode='binary', download_path=zip_file_path, url=url)

    # Extract dataset from zip file.
    with zipfile.ZipFile(zip_file_path, 'r') as input_zipfile:
      for ver in cls.vers:
        with io.TextIOWrapper(input_zipfile.open(f'wikitext-2/wiki.{ver}.tokens', 'r')) as input_binary_file:
          data = input_binary_file.read()

        with open(os.path.join(lmp.util.path.DATA_PATH, f'wiki.{ver}.tokens'), 'w') as output_text_file:
          output_text_file.write(data)

    # Remove downloaded zip file.
    os.remove(zip_file_path)
