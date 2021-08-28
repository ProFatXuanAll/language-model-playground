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
    See WikiText_ for more details on dataset.

    .. _WikiText: https://blog.einstein.ai/the-wikitext-long-term-dependency
        -language-modeling-dataset/

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
        Dataset name is ``wikitext-2``.
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
    >>> from lmp.dset import WikiText2Dset
    >>> dset = WikiText2Dset(ver='test')
    >>> dset[0][:31]
    Robert <unk> is an English film
    """
    df_ver: ClassVar[str] = 'train'
    dset_name: ClassVar[str] = 'wikitext-2'
    file_name: ClassVar[str] = 'wiki.{}.tokens.zip'
    lang: ClassVar[str] = 'en'
    vers: ClassVar[List[str]] = ['test', 'train', 'valid']
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

        # Read text file inside WikiText-2 zip file.
        with ZipFile(os.path.join(file_path), 'r') as input_zipfile:
            with TextIOWrapper(
                input_zipfile.open(f'wiki.{self.ver}.tokens', 'r'),
                encoding='utf-8',
            ) as input_text_file:
                data = input_text_file.read()

        # Remove empty line.
        spls = map(lambda spl: spl.strip(), re.split(r'\n+', data))
        spls = filter(lambda spl: spl, spls)

        # Remove section and subsection titles.
        title_pttn = re.compile(r'=.+=')
        spls = filter(lambda spl: not title_pttn.match(spl), spls)

        # Normalized dataset.
        spls = map(lmp.dset.util.norm, spls)

        # Replace unknown token `<unk>` with `[unk]`.
        unk_pttn = re.compile(r'<unk>')
        spls = map(lambda spl: unk_pttn.sub('[unk]', spl), spls)

        self.spls = list(spls)
