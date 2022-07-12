"""Dataset module.

Attributes
----------
ALL_DSETS: list[lmp.dset.BaseDset]
  All available datasets.
DSET_OPTS: typing.Final[dict[str, lmp.dset.BaseDset]]
  Mapping dataset's name ``dset_name`` to dataset's class.

Examples
--------
Get :py:class:`lmp.dset.DemoDset` by its name.

>>> from lmp.dset import DSET_OPTS, DemoDset
>>> DemoDset.dset_name in DSET_OPTS
True
>>> DSET_OPTS[DemoDset.dset_name] == DemoDset
True
"""

from typing import Dict, Final, List, Type

from lmp.dset._base import BaseDset
from lmp.dset._ch_poem import ChPoemDset
from lmp.dset._demo import DemoDset
from lmp.dset._wnli import WNLIDset
from lmp.dset._wiki_text_2 import WikiText2Dset

ALL_DSETS: Final[List[Type[BaseDset]]] = [
  ChPoemDset,
  DemoDset,
  WNLIDset,
  WikiText2Dset,
]
DSET_OPTS: Final[Dict[str, Type[BaseDset]]] = {d.dset_name: d for d in ALL_DSETS}
