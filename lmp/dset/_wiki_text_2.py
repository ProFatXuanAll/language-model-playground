"""Wiki-Text-2 dataset."""

import io
import os
import zipfile
from typing import ClassVar, List, Optional

import lmp.vars
from lmp.dset._base import BaseDset


class WikiText2Dset(BaseDset):
  """Wiki-Text-2 dataset.

  Wiki-Text-2 :footcite:`merity2017pointer` is part of the WikiText Long Term Dependency Language Modeling Dataset.
  See `Wiki-Text`_ for more details.

  .. _`Wiki-Text`:
     https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/

  Here are the statistics of each supported version.
  Tokens are separated by whitespaces.

  +-----------+-------------------+--------------------------+--------------------------+
  | Version   | Number of samples | Maximum number of tokens | Minimum number of tokens |
  +===========+===================+==========================+==========================+
  | ``test``  | 60                | 14299                    | 461                      |
  +-----------+-------------------+--------------------------+--------------------------+
  | ``train`` | 600               | 17706                    | 281                      |
  +-----------+-------------------+--------------------------+--------------------------+
  | ``valid`` | 60                | 18855                    | 778                      |
  +-----------+-------------------+--------------------------+--------------------------+

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
    CLI name of Wiki-Text-2 dataset is ``wiki-text-2``.
  spls: list[str]
    All samples in the dataset.
  ver: str
    Version of the dataset.
  vers: typing.ClassVar[list[str]]
    Supported versions including ``'train'``, ``'test'`` and ``'valid'``.

  Examples
  --------
  >>> from lmp.dset import WikiText2Dset
  >>> dset = WikiText2Dset(ver='test')
  >>> dset[0][:31]
  'Robert <unk> is an English film'
  """

  df_ver: ClassVar[str] = 'train'
  dset_name: ClassVar[str] = 'wiki-text-2'
  vers: ClassVar[List[str]] = ['test', 'train', 'valid']

  def __init__(self, *, ver: Optional[str] = None):
    super().__init__(ver=ver)

    # Make sure dataset files exist.
    self.download_dataset()

    # Read dataset from the specified version.
    # Each line is normalized.
    with open(os.path.join(lmp.vars.DATA_PATH, f'wiki.{self.ver}.tokens'), 'r') as text_file:
      lines = [self.norm(line) for line in text_file.readlines()]

    # Wiki-text-2 is consist of Wiki articles.
    # Each article is consist of one main section, many subsections and nested subsections.
    # A main section is begin with a single `=` and end with a single `=`.
    # A subsection is begin with `= =` and end with `= =`.
    # A nested subsection is begin with more than 2 `=` and end with the same amount of `=`.
    # We treat an article as one text passage.
    # Thus we loop through lines to find all sections and subsections of an article.
    article = ''
    for line_idx, line in enumerate(lines):
      # Discard empty lines.
      if not line:
        continue

      # Each article is treated as a text passage.
      # The first line is empty, so `article` is added as condition.
      # Some lines start and end with single `=` but is not section title.
      # Thus `not lines[line_idx - 1] and not lines[line_idx + 1]` is added as condition.
      if (
        article and line.startswith('=') and not line.startswith('= =') and line.endswith('=') and
        not lines[line_idx - 1] and not lines[line_idx + 1]
      ):
        # Flush previous article and start recording new article.
        self.spls.append(article.strip())
        article = line + ' '
      else:
        # Record article.
        article += line + ' '

    # Flush the last remaining article.
    self.spls.append(article)

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
    zip_file_path = os.path.join(lmp.vars.DATA_PATH, 'wiki-text-2.zip')
    # Original source is no longer available.
    url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'

    # Avoid duplicated download by checking whether all raw files exists.
    already_downloaded = True
    for ver in cls.vers:
      raw_file_path = os.path.join(lmp.vars.DATA_PATH, f'wiki.{ver}.tokens')
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

        with open(os.path.join(lmp.vars.DATA_PATH, f'wiki.{ver}.tokens'), 'w') as output_text_file:
          output_text_file.write(data)

    # Remove downloaded zip file.
    os.remove(zip_file_path)
