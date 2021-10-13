r"""WNLI dataset."""

import os
import re
from io import TextIOWrapper
from typing import ClassVar, List, Optional
from zipfile import ZipFile

import pandas as pd

import lmp.dset.util
import lmp.path
from lmp.dset._base import BaseDset


class WNLI(BaseDset):
    r"""[WNLI]_ dataset.

    WNLI is a relaxation of the Winograd Schema Challenge proposed as part of the GLUE benchmark and a conversion to the natural language inference (NLI) format.

    .. _WNLI: https://cs.nyu.edu/~davise/papers/WinogradSchemas/WS.html

    Parameters
    ==========
    ver: str, optional
        Version of the dataset.
        If ``ver is None``, then use default version ``WNLI.df_ver``
        of the dataset.
        Version must be available, available versions are

        - ``train``: Training set.
        - ``test``: Testing set.
        - ``valid``: Validation set.

        Defaults to ``None``.

    Attributes
    ==========
    df_ver: ClassVar[str]
        Default version is ``train``.
    dset_name: ClassVar[str]
        Dataset name is ``WNLI``.
        Used for command line argument parsing.
    file_name: ClassVar[str]
        Download dataset file name.
        Used only for downloading dataset files.
    lang: ClassVar[str]
        Use English as primary language.
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

    References
    ==========
    .. [WikiText-2] Stephen Merity, Caiming Xiong, James Bradbury, and
        Richard Socher. 2016. Pointer Sentinel Mixture Models

    Examples
    ========
    >>> from lmp.dset import WNLI
    >>> dset = WNLI(ver='test')
    >>> dset[0][:34]
    Maude and Dora had seen the trains
    """
    df_ver: ClassVar[str] = 'train'
    dset_name: ClassVar[str] = 'WNLI'
    file_name: ClassVar[str] = 'WNLI.zip'
    lang: ClassVar[str] = 'en'
    vers: ClassVar[List[str]] = ['test', 'train', 'dev']
    url: ClassVar[str] = ''.join([
        'https://github.com/ProFatXuanAll',
        '/demo-dataset/raw/main/wikitext-2',
    ])

    def __init__(self, *, ver: Optional[str] = None):
        super().__init__(ver=ver)

        file_path = os.path.join(
            lmp.path.DATA_PATH,
            self.__class__.file_name.format(self.ver),
        )

        # Read text file inside WNLI zip file.
        with ZipFile(os.path.join(file_path), 'r') as input_zipfile:
            with TextIOWrapper(
                input_zipfile.open(f'WNLI/{self.ver}.tsv', 'r'),
                encoding='utf-8',
            ) as input_text_file:
                df = pd.read_csv(input_text_file,sep='\t')

        # Merge tow cols.
        df["sentence"] = df["sentence1"].map(str) + df["sentence2"]

        # Normalized dataset.
        spls = df['sentence'].apply(str).apply(lmp.dset.util.norm).tolist()

        # split the letters and symbols
        spls = map(lambda spl: re.sub( r'([a-zA-Z])([,.!?])', r'\1 \2',spl), spls)
        spls = map(lambda spl: re.sub( r'([,.!?])([a-zA-Z])', r'\1 \2',spl), spls)
        
        self.spls = list(spls)
