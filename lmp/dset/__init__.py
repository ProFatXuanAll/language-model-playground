r"""Dataset module.

All dataset classes must be re-imported in this file.

Attributes
==========
ALL_DSETS: List[:py:class:`lmp.dset.BaseDset`]
    All available dataset.
    Every time a new dataset is add, it must also be added to the list.
DSET_OPTS: Final[Dict[str, :py:class:`lmp.dset.BaseDset`]]
    Mapping from dataset's name to dataset's class.
    All dataset must have class attribute ``dset_name``.
LANG_SET: Final[Set[str]]
    Set of language used in dataset.
    Only used as hint for code readability.
    All dataset must have class attribute ``lang``.

Examples
========
Check ``'wikitext-2'`` is in available dataset class.

>>> from lmp.dset import DSET_OPTS
>>> 'wikitext-2' in DSET_OPTS
True

Get ``'wikitext-2'`` dataset class.

>>> from lmp.dset import WikiText2Dset
>>> DSET_OPTS['wikitext-2'] == WikiText2Dset
True
"""

from typing import Dict, Final, Set

from lmp.dset._base_dset import BaseDset
from lmp.dset._wiki_text_2 import WikiText2Dset

ALL_DSETS = [
    WikiText2Dset,
]
DSET_OPTS: Final[Dict[str, BaseDset]] = {d.dset_name: d for d in ALL_DSETS}
LANG_SET: Final[Set[str]] = {d.lang for d in ALL_DSETS}
