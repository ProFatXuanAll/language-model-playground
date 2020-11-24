r"""Dataset base class."""


from typing import ClassVar, Iterator, List, Optional

import torch.utils.data


class BaseDset(torch.utils.data.Dataset):
    r"""Dataset base class.

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
        Used only for command line argument parsing.
    lang: ClassVar[str]
        Language of the dataset.
    spls: Sequence[str]
        All samples in the dataset.
    ver: str
        Version of the dataset.
    vers: ClassVar[List[str]]
        All supported version of the dataset.
        This is used to check whether specified version ``ver`` is supported.

    Raises
    ======
    TypeError
        When ``ver`` is not and instance of ``str``.
    ValueError
        When dataset version ``ver`` is not supported.
    """
    df_ver: ClassVar[str] = ''
    dset_name: ClassVar[str] = 'base'
    lang: ClassVar[str] = ''
    vers: ClassVar[List[str]] = []

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
                f'version {ver} is not supported.\n'
                + 'Supported version:\n'
                + ''.join(map(lambda ver: f'\t- {ver}\n', self.__class__.vers))
            )

        self.spls = []
        self.ver = ver

    def __iter__(self) -> Iterator[str]:
        r"""Iterate through each sample in the dataset.

        Yields
        ======
        str
            Each text sapmle in `self.spls`.
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
            When `idx >= len(self.spls)`.
        TypeError
            When `idx` is not an instance of `int`.
        """
        # Type check.
        if not isinstance(idx, int):
            raise TypeError('`idx` must be an instance of `int`.')

        return self.spls[idx]
