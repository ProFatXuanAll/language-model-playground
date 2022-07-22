r"""WNLI dataset."""

import io
import os
import re
import zipfile
from typing import ClassVar, List, Optional

# Typeshed for `pandas` is under development, we will ignore type check on `pandas` until `pandas` typeshed finish its
# development and release stable version.
import pandas as pd  # type: ignore

import lmp.util.path
from lmp.dset._base import BaseDset


class WNLIDset(BaseDset):
  """Winograd NLI dataset.

  Winograd NLI [1]_ is a relaxation of the `Winograd Schema Challenge`_ proposed as part of the GLUE_ benchmark.
  This dataset only extract sentences from WNLI and no NLI labels were used.

  .. _`Winograd Schema Challenge`: https://cs.nyu.edu/~davise/papers/WinogradSchemas/WS.html
  .. _GLUE: https://gluebenchmark.com/

  Here are the statistics of each supported version.
  Tokens are separated by whitespaces.

  +-----------+-------------------+--------------------------+--------------------------+
  | Version   | Number of samples | Maximum number of tokens | Minimum number of tokens |
  +===========+===================+==========================+==========================+
  | ``dev``   | 142               | 63                       | 4                        |
  +-----------+-------------------+--------------------------+--------------------------+
  | ``test``  | 292               | 60                       | 4                        |
  +-----------+-------------------+--------------------------+--------------------------+
  | ``train`` | 1270              | 63                       | 3                        |
  +-----------+-------------------+--------------------------+--------------------------+

  Parameters
  ----------
  ver: Optional[str], default: None
    Version of the dataset.
    Set ``ver = ''`` to use default version.

  Attributes
  ----------
  df_ver: typing.ClassVar[str]
    Default version is ``train``.
  dset_name: typing.ClassVar[str]
    CLI name of WNLI dataset is ``WNLI``.
  spls: list[str]
    All samples in the dataset.
  ver: str
    Version of the dataset.
  vers: typing.ClassVar[list[str]]
    Supported versions including ``'train'``, ``'dev'`` and ``'test'``.

  References
  ----------
  .. [1] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy and Samuel R. Bowman.  GLUE: A multi-task
     benchmark and analysis platform for natural language understanding.  ICLR 2019.

  Examples
  --------
  >>> from lmp.dset import WNLIDset
  >>> dset = WNLIDset(ver='test')
  >>> dset[0]
  Mark was timid .
  """

  df_ver: ClassVar[str] = 'train'
  dset_name: ClassVar[str] = 'WNLI'
  vers: ClassVar[List[str]] = ['dev', 'test', 'train']

  def __init__(self, *, ver: Optional[str] = None):
    super().__init__(ver=ver)

    # Make sure dataset files exist.
    self.download_dataset()

    # Read text from WNLI tsv file.
    df = pd.read_csv(os.path.join(lmp.util.path.DATA_PATH, f'wnli.{self.ver}.tsv'), sep='\t')

    # Extract all sentences and perform text normalization.
    spls = df['sentence1'].apply(self.norm).tolist() + df['sentence2'].apply(self.norm).tolist()

    # Insert space before punctuation marks and abbreviations.
    spls = list(map(lambda spl: re.sub(r'(\w)([,.!?:;"\'-])', r'\1 \2', spl), spls))
    spls = list(map(lambda spl: re.sub(r'(["])(\w)', r'\1 \2', spl), spls))
    spls = list(map(lambda spl: re.sub(r'(\w)(\'\w)\s+', r'\1 \2 ', spl), spls))

    self.spls.extend(spls)

    # Sort dataset by length in ascending order.
    self.spls.sort(key=len)

  @classmethod
  def download_dataset(cls) -> None:
    """Download WNLI dataset.

    Download zip file from https://dl.fbaipublicfiles.com/glue/data/WNLI.zip and extract raw files from zip file.
    Raw files are named as ``wnli.ver.tsv``, where ``ver`` is the version of the dataset.
    After extracting raw files the downloaded zip file will be deleted.

    Returns
    -------
    None
    """
    # Download zip file path.
    zip_file_path = os.path.join(lmp.util.path.DATA_PATH, 'WNLI.zip')
    # Original source is no longer available.
    url = 'https://dl.fbaipublicfiles.com/glue/data/WNLI.zip'

    # Avoid duplicated download by checking whether all raw files exists.
    already_downloaded = True
    for ver in cls.vers:
      raw_file_path = os.path.join(lmp.util.path.DATA_PATH, f'wnli.{ver}')
      if not os.path.exists(raw_file_path):
        already_downloaded = False

    if already_downloaded:
      return

    # Download dataset.
    BaseDset.download_file(mode='binary', download_path=zip_file_path, url=url)

    # Extract dataset from zip file.
    with zipfile.ZipFile(zip_file_path, 'r') as input_zipfile:
      for ver in cls.vers:
        with io.TextIOWrapper(input_zipfile.open(f'WNLI/{ver}.tsv', 'r')) as input_binary_file:
          data = input_binary_file.read()

        with open(os.path.join(lmp.util.path.DATA_PATH, f'wnli.{ver}.tsv'), 'w') as output_text_file:
          output_text_file.write(data)

    # Remove downloaded zip file.
    os.remove(zip_file_path)
