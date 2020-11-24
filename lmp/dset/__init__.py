from typing import Final, FrozenSet, Set

from lmp.dset._base_dset import BaseDset
from lmp.dset._wiki_text_2 import WikiText2Dset

LANG_SET: Final[FrozenSet[str]] = frozenset({'en', 'zh'})
