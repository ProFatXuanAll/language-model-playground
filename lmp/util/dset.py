r"""Dataset utilities."""

from typing import Optional

from lmp.dset import DSET_OPTS, BaseDset


def load(dset_name: str, ver: Optional[str] = None) -> BaseDset:
    r"""Load dataset.

    Parameters
    ==========
    dset_name: str
        Name of the dataset to load.
    ver: str, optional
        Version of the dataset to load.
        Load default version of the dataset if set to ``None``.
        Defaults to ``None``.

    Returns
    =======
    lmp.dset.BaseDset
        Loaded dataset instance.

    See Also
    ========
    lmp.dset
        All available dataset.

    Examples
    ========
    >>> from lmp.dset import WikiText2Dset
    >>> import lmp.util.dset
    >>> isinstance(lmp.util.dset.load('wikitext2'), WikiText2Dset)
    True
    """
    return DSET_OPTS[dset_name](ver=ver)
