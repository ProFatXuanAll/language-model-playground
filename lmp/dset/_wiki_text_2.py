r"""WikiText-2 Dataset."""

import io
import os
import zipfile
from typing import ClassVar, List, Optional

import lmp.path
from lmp.dset._base_dset import BaseDset


class WikiText2Dset(BaseDset):
    r"""WikiText-2 Dataset.

    WikiText-2 is an English dataset which is part of the WikiText Long Term
    Dependency Language Modeling Dataset.
    See https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/
    for more details on dataset.

    Parameters
    ==========
    ver: str, optional
        Version of the dataset.
        If ``ver is None``, then use default version ``self.__class__.df_ver`
        of the dataset.
        Version must be supported by the dataset, see ``self.__class__.vers``
        for list of supported versions.

    Attributes
    ==========
    dset_name: ClassVar[str]
        Display name for dataset on CLI.
        Used only for command line argument parsing.
    lang: ClassVar[str]
        Language of the dataset.
    spls: Sequence[str]
        All samples in the dataset.
    ver: str
        Version of the dataset.

    Raises
    ======
    TypeError
        When ``ver`` is not and instance of ``str``.
    ValueError
        When dataset version ``ver`` is not supported.
    """
    df_ver: ClassVar[str] = 'train'
    dset_name: ClassVar[str] = 'wikitext-2'
    lang: ClassVar[str] = 'en'
    vers: ClassVar[List[str]] = ['test', 'train', 'valid']

    def __init__(self, *, ver: Optional[str] = None):
        super().__init__(ver=ver)

        # Read text file inside WikiText-2 zip file.
        with zipfile.ZipFile(
            os.path.join(lmp.path.DATA_PATH, 'wikitext-2-v1.zip'),
            'r',
        ) as zf:
            with io.TextIOWrapper(
                zf.open(f'wikitext-2/wiki.{self.ver}.tokens', 'r'),
                encoding='utf-8',
            ) as input_file:
                spls = input_file.readlines()

        self.spls = spls
