r"""WikiText-2 dataset."""

import os
import re
from io import TextIOWrapper
from typing import ClassVar, List, Optional
from zipfile import ZipFile

import lmp.dset.util
import lmp.path
from lmp.dset._base import BaseDset


class WikiText2Dset(BaseDset):
    r"""[WikiText-2]_ dataset.

    WikiText-2 is an English dataset which is part of the WikiText Long Term
    Dependency Language Modeling Dataset.
    See https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/
    for more details on dataset.

    Parameters
    ==========
    ver: str, optional
        Version of the dataset.
        If ``ver is None``, then use default version ``WikiText2Dset.df_ver``
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
        Display name for dataset on CLI.
        Used only for command line argument parsing.
    lang: ClassVar[str]
        Use English as primary language.
    spls: Sequence[str]
        All samples in the dataset.
    ver: str
        Version of the dataset.
    vers: ClassVar[List[str]]
        All available version of the dataset.
        Used to check whether specified version ``ver`` is available.

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
    >>> from lmp.dset import WikiText2Dset
    >>> dset = WikiText2Dset(ver='test')
    >>> dset[0][:31]
    Robert <unk> is an English film
    """
    df_ver: ClassVar[str] = 'train'
    dset_name: ClassVar[str] = 'wikitext-2'
    lang: ClassVar[str] = 'en'
    vers: ClassVar[List[str]] = ['test', 'train', 'valid']

    def __init__(self, *, ver: Optional[str] = None):
        super().__init__(ver=ver)

        # Read text file inside WikiText-2 zip file.
        with ZipFile(
            os.path.join(lmp.path.DATA_PATH, 'wikitext-2-v1.zip'),
            'r',
        ) as input_zipfile:
            with TextIOWrapper(
                input_zipfile.open(os.path.join('wikitext-2', f'wiki.{self.ver}.tokens'), 'r'),
                encoding='utf-8',
            ) as input_text_file:
                data = input_text_file.read()

        # Remove empty line.
        spls = filter(lambda spl: spl.strip(), re.split(r'\n', data))
        # Remove section and subsection titles.
        pttn = re.compile(r'( =){1,3} .+ (= ){1,3}')
        spls = filter(lambda spl: not pttn.match(spl), spls)
        # Normalized dataset.
        spls = list(map(lmp.dset.util.norm, spls))

        self.spls = spls
