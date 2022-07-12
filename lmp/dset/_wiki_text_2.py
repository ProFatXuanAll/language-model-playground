"""Wiki-Text-2 dataset."""

import io
import os
import re
import zipfile
from typing import ClassVar, List, Optional

import lmp.util.path
from lmp.dset._base import BaseDset
from lmp.tknzr._base import UNK_TK


class WikiText2Dset(BaseDset):
  """Wiki-Text-2 dataset.

  Wiki-Text-2 [1]_ is part of the WikiText Long Term Dependency Language Modeling Dataset.
  See WikiText_ for more details on dataset.

  .. _WikiText:
     https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/

  Here are the statistics of each supported version.
  Tokens are separated by whitespaces.

  +-----------+-------------------+--------------------------+--------------------------+
  | Version   | Number of samples | Maximum number of tokens | Minimum number of tokens |
  +===========+===================+==========================+==========================+
  | ``train`` | 14628             | 699                      | 10                       |
  +-----------+-------------------+--------------------------+--------------------------+
  | ``test``  | 1718              | 481                      | 10                       |
  +-----------+-------------------+--------------------------+--------------------------+
  | ``valid`` | 1533              | 429                      | 10                       |
  +-----------+-------------------+--------------------------+--------------------------+

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
    CLI name of Wiki-Text-2 dataset is ``wiki-text-2``.
  spls: list[str]
    All samples in the dataset.
  ver: str
    Version of the dataset.
  vers: typing.ClassVar[list[str]]
    Supported versions including ``'train'``, ``'test'`` and ``'valid'``.

  References
  ----------
  .. [1] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2016. Pointer Sentinel Mixture Models

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

    for line in lines:
      # Perform text normalization.
      line = self.norm(line)

      # Discard empty lines.
      if not line:
        continue
      # Discard section titles.
      if line.startswith('=') and line.endswith('='):
        continue
      # Discard samples consist mainly non-alphabetic words.
      if len(re.findall(r'(<formula>|<unk>|[^a-zA-Z])', line)) / len(line) >= 0.3:
        continue
      # Discard samples having fewer than 10 words.
      if len(line.split(' ')) < 10:
        continue

      # Replace `@.@` token with middle character.
      line = re.sub(r'@(.)@', r'\1', line)

      # Replace unknown token `<unk>` with `[unk]`.
      line = re.sub(r'<unk>', UNK_TK, line)

      # Add the preprocessed line to dataset.
      self.spls.append(line)

    # Sort dataset by length in ascending order.
    self.spls.sort(key=len)

  @classmethod
  def download_dataset(cls) -> None:
    """Download Wiki-text-2 dataset.

    Download zip file from https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip and extract raw
    files from zip file.
    Raw files are named as ``wiki.ver.tokens``, where ``ver`` is the version of the dataset.
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
