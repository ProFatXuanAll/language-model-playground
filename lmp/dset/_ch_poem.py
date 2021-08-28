r"""Chinese poetry dataset."""

import os
from io import TextIOWrapper
from typing import ClassVar, List, Optional
from zipfile import ZipFile

import pandas as pd

import lmp.dset.util
import lmp.path
from lmp.dset._base import BaseDset


class ChPoemDset(BaseDset):
    r"""Chinese poem dataset.

    Collection of poems dating way back to ancient Chinese dynasty.
    See https://github.com/Werneror/Poetry for more details on dataset.

    Some poems are preprocessed as follow:

    - Combine scattered files into one (including ``宋``, ``明``, ``清``)
    - Remove empty content (with value ``無正文。``)

    Parameters
    ==========
    ver: str, optional
        Version of the dataset.
        If ``ver is None``, then use default version ``ChPoemDset.df_ver`` of
        the dataset.
        Version must be available.
        Available versions are named after their appearing time, including
        ``元``, ``元末明初``, ``先秦``, ``南北朝``, ``唐``, ``唐末宋初``, ``宋``, ``宋末元初``,
        ``宋末金初``, ``明``, ``明末清初``, ``民國末當代初``, ``清``, ``清末民國初``, ``清末近現代初``,
        ``漢``, ``當代``, ``秦``, ``近現代``, ``近現代末當代初``, ``遼``, ``金``, ``金末元初``,
        ``隋``, ``隋末唐初``, ``魏晉``, ``魏晉末南北朝初``.

        Defaults to ``None``.

    Attributes
    ==========
    df_ver: ClassVar[str]
        Default version is ``唐``.
    dset_name: ClassVar[str]
        Dataset name is ``chinese-poem``.
        Used for command line argument parsing.
    file_name: ClassVar[str]
        Download dataset file name.
        Used only for downloading dataset files.
    lang: ClassVar[str]
        Use Chinese as primary language.
    spls: List[str]
        All samples in the dataset.
    ver: str
        Version of the dataset.
    vers: ClassVar[List[str]]
        All available versions of the dataset.
        Used to check whether specified version ``ver`` is available.
    url: ClassVar[str]
        URL for downloading dataset files.
        Used only for downloading dataset files.

    Raises
    ======
    TypeError
        When ``ver`` is not and instance of ``str``.
    ValueError
        When dataset version ``ver`` is not available.

    See Also
    ========
    lmp.dset.BaseDset

    Examples
    ========
    >>> from lmp.dset import ChPoemDset
    >>> dset = ChPoemDset(ver='唐')
    >>> dset[0][:10]
    風淅淅。夜雨連雲黑。
    """
    df_ver: ClassVar[str] = '唐'
    dset_name: ClassVar[str] = 'chinese-poem'
    file_name: ClassVar[str] = '{}.csv.zip'
    lang: ClassVar[str] = 'zh'
    vers: ClassVar[List[str]] = [
        '元', '元末明初', '先秦', '南北朝', '唐', '唐末宋初', '宋', '宋末元初', '宋末金初', '明',
        '明末清初', '民國末當代初', '清', '清末民國初', '清末近現代初', '漢', '當代', '秦', '近現代',
        '近現代末當代初', '遼', '金', '金末元初', '隋', '隋末唐初', '魏晉', '魏晉末南北朝初',
    ]
    url: ClassVar[str] = ''.join([
        'https://github.com/ProFatXuanAll',
        '/demo-dataset/raw/main/ch-poem',
    ])

    def __init__(self, *, ver: Optional[str] = None):
        super().__init__(ver=ver)

        file_path = os.path.join(
            lmp.path.DATA_PATH,
            self.__class__.file_name.format(self.ver),
        )

        # Read text file inside chinese poem zip file.
        with ZipFile(os.path.join(file_path), 'r') as input_zipfile:
            with TextIOWrapper(
                input_zipfile.open(f'{self.ver}.csv', 'r'),
                encoding='utf-8',
            ) as input_text_file:
                df = pd.read_csv(input_text_file)

        # Normalized dataset.
        spls = df['內容'].apply(str).apply(lmp.dset.util.norm).tolist()

        self.spls = spls
