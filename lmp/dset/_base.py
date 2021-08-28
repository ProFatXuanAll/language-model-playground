r"""Dataset base class."""

import os
from typing import ClassVar, Iterator, List, Optional

import torch.utils.data

import lmp.dset.util
import lmp.path


class BaseDset(torch.utils.data.Dataset):
    r"""Dataset base class.

    All dataset files are hosted on `demo-dataset`_ repository.
    If any dataset files is not on your local machine, then
    :py:class:`lmp.dset.BaseDset` will automatically download dataset files
    from `demo-dataset`_ repository.
    Once dataset files are downloaded, they will not be downloaded again.

    .. _`demo-dataset`: https://github.com/ProFatXuanAll/demo-dataset

    Parameters
    ==========
    ver: str, optional
        Version of the dataset.
        If ``ver is None``, then use default version ``self.__class__.df_ver``
        of the dataset.

    Attributes
    ==========
    df_ver: ClassVar[str]
        Default version of the dataset.
    dset_name: ClassVar[str]
        Display name for dataset on CLI.
        Used for command line argument parsing.
        Subclass must overwrite ``dset_name`` attribute.
    file_name: ClassVar[str]
        Download dataset file name.
        Used only for downloading dataset files.
    lang: ClassVar[str]
        Language of the dataset.
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
    """
    df_ver: ClassVar[str] = ''
    dset_name: ClassVar[str] = 'base'
    file_name: ClassVar[str] = ''
    lang: ClassVar[str] = ''
    vers: ClassVar[List[str]] = []
    url: ClassVar[str] = ''

    def __init__(self, *, ver: Optional[str] = None):
        super().__init__()
        # Use default version of the dataset.
        if ver is None:
            ver = self.__class__.df_ver
        # Type check.
        elif not isinstance(ver, str):
            raise TypeError('`ver` must be an instance of `str`.')
        elif ver not in self.__class__.vers:
            raise ValueError(
                f'Version {ver} is not available.\n'
                + 'Available versions:\n'
                + ''.join(map(lambda ver: f'\t- {ver}\n', self.__class__.vers))
            )

        self.spls: List[str] = []
        self.ver = ver

        # Download dataset file if file does not exist.
        self.download()

    def __iter__(self) -> Iterator[str]:
        r"""Iterate through each sample in the dataset.

        Yields
        ======
        str
            Each text sapmle in ``self.spls``.
        """
        for spl in self.spls:
            yield spl

    def __len__(self) -> int:
        r"""Get dataset size.

        Returns
        =======
        int
            Number of samples in the dataset.
        """
        return len(self.spls)

    def __getitem__(self, idx: int) -> str:
        r"""Sample text using index.

        Parameters
        ==========
        idx: int
            Sample index.

        Raises
        ======
        IndexError
            When ``idx >= len(self.spls)``.
        TypeError
            When ``idx`` is not an instance of ``int``.
        """
        # Type check.
        if not isinstance(idx, int):
            raise TypeError('`idx` must be an instance of `int`.')

        return self.spls[idx]

    def download(self) -> None:
        r"""Download dataset file if not exists.

        Only download dataset file if not exists.
        Once dataset is downloaded it will not be downloaded again.
        """
        file_name = self.__class__.file_name.format(self.ver)
        file_path = os.path.join(lmp.path.DATA_PATH, file_name)

        # Cancel download if dataset file already existed.
        if os.path.exists(file_path):
            return

        # Download dataset file.
        url = f'{self.__class__.url}/{file_name}'
        lmp.dset.util.download(url=url, file_path=file_path)
