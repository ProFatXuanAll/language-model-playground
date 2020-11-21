r"""Tokenizer module.

All tokenizer must import from this file.

Example
=======

::

    import lmp.tknzr

    tknzr = lmp.tknzr.CharTknzr(...)
    tknzr = lmp.tknzr.WsTknzr(...)
"""

import lmp.tknzr.util
from lmp.tknzr._base_tknzr import BaseTknzr
from lmp.tknzr._char_tknzr import CharTknzr
from lmp.tknzr._ws_tknzr import WsTknzr
