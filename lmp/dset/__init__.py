"""Dataset module.

Attributes
----------
ALL_DSETS: list[lmp.dset.BaseDset]
  All available dataset.
DSET_OPTS: typing.Final[dict[str, lmp.dset.BaseDset]]
  Mapping dataset's name ``dset_name`` to dataset's class.

Examples
--------
Get :py:class:`lmp.dset.WikiText2Dset` by its name.

>>> from lmp.dset import DSET_OPTS, WikiText2Dset
>>> WikiText2Dset.dset_name in DSET_OPTS
True
>>> DSET_OPTS[WikiText2Dset.dset_name] == WikiText2Dset
True
"""

from typing import Dict, Final, List, Type

from lmp.dset._base import BaseDset
from lmp.dset._ch_poem import ChPoemDset
from lmp.dset._wiki_text_2 import WikiText2Dset

ALL_DSETS: Final[List[Type[BaseDset]]] = [
  ChPoemDset,
  WikiText2Dset,
]
DSET_OPTS: Final[Dict[str, Type[BaseDset]]] = {d.dset_name: d for d in ALL_DSETS}
