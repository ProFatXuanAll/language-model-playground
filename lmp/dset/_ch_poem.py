"""Chinese poetry dataset."""

import io
import os
import zipfile
from typing import ClassVar, List, Optional

# Typeshed for `pandas` is under development, we will ignore type check on `pandas` until `pandas` typeshed finish its
# development and release stable version.
import pandas as pd  # type: ignore

import lmp.util.path
from lmp.dset._base import BaseDset


class ChPoemDset(BaseDset):
  """Chinese poem dataset.

  Poems of ancient Chinese dynasty.  See https://github.com/Werneror/Poetry for details on dataset.  See
  https://github.com/ProFatXuanAll/demo-dataset for dataset preprocessing details.

  Here we list some dataset statistics.

  +--------------------+-----------------+-------------------+
  | dynasty            | number of poems | number of authors |
  +====================+=================+===================+
  | ``宋``             | 287114          | 9446              |
  +--------------------+-----------------+-------------------+
  | ``明``             | 236957          | 4439              |
  +--------------------+-----------------+-------------------+
  | ``清``             | 90089           | 8872              |
  +--------------------+-----------------+-------------------+
  | ``唐``             | 49195           | 2736              |
  +--------------------+-----------------+-------------------+
  | ``元``             | 37375           | 1209              |
  +--------------------+-----------------+-------------------+
  | ``近現代``         | 28419           | 790               |
  +--------------------+-----------------+-------------------+
  | ``當代``           | 28219           | 177               |
  +--------------------+-----------------+-------------------+
  | ``明末清初``       | 17700           | 176               |
  +--------------------+-----------------+-------------------+
  | ``元末明初``       | 15736           | 79                |
  +--------------------+-----------------+-------------------+
  | ``清末民國初``     | 15367           | 99                |
  +--------------------+-----------------+-------------------+
  | ``清末近現代初``   | 12464           | 48                |
  +--------------------+-----------------+-------------------+
  | ``宋末元初``       | 12058           | 41                |
  +--------------------+-----------------+-------------------+
  | ``南北朝``         | 4586            | 434               |
  +--------------------+-----------------+-------------------+
  | ``近現代末當代初`` | 3426            | 23                |
  +--------------------+-----------------+-------------------+
  | ``魏晉``           | 3020            | 251               |
  +--------------------+-----------------+-------------------+
  | ``金末元初``       | 3019            | 17                |
  +--------------------+-----------------+-------------------+
  | ``金``             | 2741            | 253               |
  +--------------------+-----------------+-------------------+
  | ``民國末當代初``   | 1948            | 9                 |
  +--------------------+-----------------+-------------------+
  | ``隋``             | 1170            | 84                |
  +--------------------+-----------------+-------------------+
  | ``唐末宋初``       | 1118            | 44                |
  +--------------------+-----------------+-------------------+
  | ``先秦``           | 570             | 8                 |
  +--------------------+-----------------+-------------------+
  | ``隋末唐初``       | 472             | 40                |
  +--------------------+-----------------+-------------------+
  | ``漢``             | 363             | 83                |
  +--------------------+-----------------+-------------------+
  | ``宋末金初``       | 234             | 9                 |
  +--------------------+-----------------+-------------------+
  | ``遼``             | 22              | 7                 |
  +--------------------+-----------------+-------------------+
  | ``秦``             | 2               | 2                 |
  +--------------------+-----------------+-------------------+
  | ``魏晉末南北朝初`` | 1               | 1                 |
  +--------------------+-----------------+-------------------+
  | total              | 853385          | 29377             |
  +--------------------+-----------------+-------------------+

  Parameters
  ----------
  ver: Optional[str], default: None
    Version of the dataset.   Set ``ver = ''`` to use default version.

  Attributes
  ----------
  df_ver: typing.ClassVar[str]
    Default version is ``'唐'``.
  dset_name: typing.ClassVar[str]
    Chinese poem dataset's name is ``chinese-poem``.
  spls: list[str]
    All samples in the dataset.
  ver: str
    Version of the dataset.
  vers: typing.ClassVar[list[str]]
    All available versions of the dataset.  Versions are named after their appearing times, including ``元``,
    ``元末明初``, ``先秦``, ``南北朝``, ``唐``, ``唐末宋初``, ``宋``, ``宋末元初``, ``宋末金初``, ``明``, ``明末清初``,
    ``民國末當代初``, ``清``, ``清末民國初``, ``清末近現代初``, ``漢``, ``當代``, ``秦``, ``近現代``,
    ``近現代末當代初``, ``遼``, ``金``, ``金末元初``, ``隋``, ``隋末唐初``, ``魏晉``, ``魏晉末南北朝初``.

  See Also
  --------
  lmp.dset
    All available datasets.
  lmp.dset.BaseDset
    Dataset utilities.

  Examples
  --------
  >>> from lmp.dset import ChPoemDset
  >>> dset = ChPoemDset(ver='唐')
  >>> dset[0][:10]
  '風淅淅。夜雨連雲黑。'
  """

  df_ver: ClassVar[str] = '唐'
  dset_name: ClassVar[str] = 'chinese-poem'
  vers: ClassVar[List[str]] = [
    '元',
    '元末明初',
    '先秦',
    '南北朝',
    '唐',
    '唐末宋初',
    '宋',
    '宋末元初',
    '宋末金初',
    '明',
    '明末清初',
    '民國末當代初',
    '清',
    '清末民國初',
    '清末近現代初',
    '漢',
    '當代',
    '秦',
    '近現代',
    '近現代末當代初',
    '遼',
    '金',
    '金末元初',
    '隋',
    '隋末唐初',
    '魏晉',
    '魏晉末南北朝初',
  ]

  def __init__(self, *, ver: Optional[str] = None):
    super().__init__(ver=ver)

    # Make sure dataset files exist.
    self.download_dataset(ver=self.ver)

    # Read text file inside chinese poem zip file.
    df = pd.read_csv(os.path.join(lmp.util.path.DATA_PATH, f'{self.ver}.csv'))

    # Normalize dataset.
    self.spls = df['內容'].apply(str).apply(self.norm).tolist()

  @classmethod
  def download_dataset(cls, ver: str) -> None:
    """Download Chinese poem dataset.

    Download zip file from https://github.com/ProFatXuanAll/demo-dataset/raw/main/ch-poem and extract raw file from
    zip file.  Raw files are named as ``'ver.csv'``, where ``ver`` is the version of the dataset.  After extracting raw
    files the downloaded zip file will be deleted.

    Parameters
    ----------
    ver: str
      Version of the dataset.

    Returns
    -------
    None
    """
    # `ver` validation.
    lmp.util.validate.raise_if_not_instance(val=ver, val_name='ver', val_type=str)
    lmp.util.validate.raise_if_not_in(val=ver, val_name='ver', val_range=cls.vers)

    # Download zip file path.
    zip_file_path = os.path.join(lmp.util.path.DATA_PATH, f'{ver}.csv.zip')
    # We host this dataset on GitHub.
    url = f'https://github.com/ProFatXuanAll/demo-dataset/raw/main/ch-poem/{ver}.csv.zip'

    # Avoid duplicated download by checking whether raw file exists.
    raw_file_path = os.path.join(lmp.util.path.DATA_PATH, f'{ver}.csv')
    if os.path.exists(raw_file_path):
      return

    # Download dataset.
    BaseDset.download_file(mode='binary', download_path=zip_file_path, url=url)

    # Extract dataset from zip file.
    with zipfile.ZipFile(os.path.join(zip_file_path), 'r') as input_zipfile:
      with io.TextIOWrapper(input_zipfile.open(f'{ver}.csv', 'r'), encoding='utf-8') as input_binary_file:
        data = input_binary_file.read()
      with open(os.path.join(lmp.util.path.DATA_PATH, f'{ver}.csv'), 'w') as output_text_file:
        output_text_file.write(data)

    # Remove downloaded zip file.
    os.remove(zip_file_path)
