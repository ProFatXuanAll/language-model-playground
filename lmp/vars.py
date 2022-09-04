"""Variables shared throughout this project.

Attributes
----------
BOS_TK: typing.Final[str]
  A special token which represents the begining of a sequence.
BOS_TKID: typing.Final[int]
  Token id of :py:attr:`lmp.vars.BOS_TK`.
DATA_PATH: Final[str]
  Absolute path of all the dataset.
  Some datasets are hosted on remote servers, thus this variable serve as the dataset download location.
  See :py:class:`lmp.dset.BaseDset` for more information.
EOS_TK: typing.Final[str]
  A special token which represents the end of a sequence.
EOS_TKID: typing.Final[int]
  Token id of :py:attr:`lmp.vars.EOS_TK`.
EXP_PATH: Final[str]
  Absolute path of all experiments.
  Experiments are ignored by git.
  No experiment results (model checkpoints, tokenizer cofiguration, etc.) will be commited.
LOG_PATH: Final[str]
  Absolute path of all experiments' log.
  Experiments are ignored by git.
  No experiment logs will be commited.
PAD_TK: typing.Final[str]
  A special token which represents paddings of a sequence.
PAD_TKID: typing.Final[int]
  Token id of :py:attr:`lmp.vars.PAD_TK`.
PROJECT_ROOT: Final[str]
  Absolute path of the project root directory.
SP_TKS: typing.Final[list[str]]
  List of special tokens.
UNK_TK: typing.Final[str]
  A special token which represents unknown tokens in a sequence.
UNK_TKID: typing.Final[int]
  Token id of :py:attr:`lmp.vars.UNK_TK`.

Example
-------
>>> import lmp.vars
>>> assert isinstance(lmp.vars.BOS_TK, str)
>>> assert isinstance(lmp.vars.BOS_TKID, int)
>>> assert isinstance(lmp.vars.DATA_PATH, str)
>>> assert isinstance(lmp.vars.EOS_TK, str)
>>> assert isinstance(lmp.vars.EOS_TKID, int)
>>> assert isinstance(lmp.vars.EXP_PATH, str)
>>> assert isinstance(lmp.vars.LOG_PATH, str)
>>> assert isinstance(lmp.vars.PAD_TK, str)
>>> assert isinstance(lmp.vars.PAD_TKID, int)
>>> assert isinstance(lmp.vars.PROJECT_ROOT, str)
>>> assert isinstance(lmp.vars.SP_TKS, list)
>>> assert lmp.vars.SP_TKS == [lmp.vars.BOS_TK, lmp.vars.EOS_TK, lmp.vars.PAD_TK, lmp.vars.UNK_TK]
>>> assert isinstance(lmp.vars.UNK_TK, str)
>>> assert isinstance(lmp.vars.UNK_TKID, int)
"""

import os
from typing import Final, List

BOS_TK: Final[str] = '<bos>'
BOS_TKID: Final[int] = 0
EOS_TK: Final[str] = '<eos>'
EOS_TKID: Final[int] = 1
PAD_TK: Final[str] = '<pad>'
PAD_TKID: Final[int] = 2
UNK_TK: Final[str] = '<unk>'
UNK_TKID: Final[int] = 3
SP_TKS: Final[List[str]] = [BOS_TK, EOS_TK, PAD_TK, UNK_TK]

PROJECT_ROOT: Final[str] = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir, os.pardir))
DATA_PATH: Final[str] = os.path.join(PROJECT_ROOT, 'data')
EXP_PATH: Final[str] = os.path.join(PROJECT_ROOT, 'exp')
LOG_PATH: Final[str] = os.path.join(EXP_PATH, 'log')
