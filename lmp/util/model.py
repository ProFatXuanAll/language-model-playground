"""Model utilities."""

import os
import re
from typing import Any, List

import torch

import lmp.util.validate
import lmp.vars
from lmp.model import MODEL_OPTS, BaseModel


def create(model_name: str, **kwargs: Any) -> BaseModel:
  """Create language model instance by language model's name.

  Language model's arguments are collected in ``**kwargs`` and are passed directly to language model's constructor.

  Parameters
  ----------
  model_name: str
    Name of the language model to create.
  kwargs: typing.Any, optional
    Model's hyperparameters.

  Returns
  -------
  ~lmp.model.BaseModel
    Language model instance.

  See Also
  --------
  :doc:`lmp.model </model/index>`
    All available language models.
  :doc:`lmp.tknzr </tknzr/index>`
    All available tokenizers.

  Examples
  --------
  >>> from lmp.model import ElmanNet
  >>> from lmp.tknzr import CharTknzr
  >>> import lmp.util.model
  >>> tknzr = CharTknzr()
  >>> model = lmp.util.model.create(model_name=ElmanNet.model_name, tknzr=tknzr)
  >>> assert isinstance(model, ElmanNet)
  """
  # `model_name` validation.
  lmp.util.validate.raise_if_not_instance(val=model_name, val_name='model_name', val_type=str)
  lmp.util.validate.raise_if_not_in(val=model_name, val_name='model_name', val_range=list(MODEL_OPTS.keys()))

  return MODEL_OPTS[model_name](**kwargs)


def save(ckpt: int, exp_name: str, model: BaseModel) -> None:
  """Save model checkpoint.

  .. danger::

    This method overwrite existing files.
    Make sure you know what you are doing before calling this method.

  Parameters
  ----------
  ckpt: int
    Saving checkpoint number.
  exp_name: int
    Language model training experiment name.
  model: lmp.model.BaseModel
    Model to be saved.

  Returns
  -------
  None

  See Also
  --------
  ~load
    Load pre-trained language model instance by checkpoint and experiment name.

  Examples
  --------
  >>> from lmp.model import ElmanNet
  >>> from lmp.tknzr import CharTknzr
  >>> import lmp.util.model
  >>> tknzr = CharTknzr()
  >>> model = ElmanNet(tknzr=tknzr)
  >>> lmp.util.model.save(ckpt=0, exp_name='test', model=model)
  None
  """
  # `ckpt` validation.
  lmp.util.validate.raise_if_not_instance(val=ckpt, val_name='ckpt', val_type=int)
  lmp.util.validate.raise_if_wrong_ordered(vals=[0, ckpt], val_names=['0', 'ckpt'])

  # `exp_name` validation.
  lmp.util.validate.raise_if_not_instance(val=exp_name, val_name='exp_name', val_type=str)
  lmp.util.validate.raise_if_empty_str(val=exp_name, val_name='exp_name')

  # `save_dir_path` validation
  save_dir_path = os.path.join(lmp.vars.EXP_PATH, exp_name)
  lmp.util.validate.raise_if_is_file(path=save_dir_path)

  if not os.path.exists(save_dir_path):
    os.makedirs(save_dir_path)

  # `save_path` validation.
  save_file_path = os.path.join(save_dir_path, f'model-{ckpt}.pt')
  lmp.util.validate.raise_if_is_directory(path=save_file_path)

  # Save model.
  torch.save(model, save_file_path)


def load(ckpt: int, exp_name: str) -> BaseModel:
  """Load pre-trained language model instance by checkpoint and experiment name.

  Load pre-trained language model from path ``project_root/exp/exp_name``.

  Parameters
  ----------
  ckpt: int
    Saving checkpoint number.
    Set to ``-1`` to load the last checkpoint.
  exp_name: str
    Pre-trained language model experiment name.

  Returns
  -------
  ~lmp.model.BaseModel
    Pre-trained language model instance.

  See Also
  --------
  :doc:`lmp.model </model/index>`
    All available language models.

  Examples
  --------
  >>> from lmp.model import ElmanNet
  >>> from lmp.tknzr import CharTknzr
  >>> import lmp.util.model
  >>> tknzr = CharTknzr()
  >>> model = ElmanNet(tknzr=tknzr)
  >>> lmp.util.model.save(ckpt=0, exp_name='test', model=model)
  >>> load_model = lmp.util.model.load(ckpt=0, exp_name='test')
  >>> assert torch.all(load_model.emb.weight == model.emb.weight)
  """
  # `ckpt` validation.
  lmp.util.validate.raise_if_not_instance(val=ckpt, val_name='ckpt', val_type=int)
  lmp.util.validate.raise_if_wrong_ordered(vals=[-1, ckpt], val_names=['-1', 'ckpt'])

  # `exp_name` validation.
  lmp.util.validate.raise_if_not_instance(val=exp_name, val_name='exp_name', val_type=str)
  lmp.util.validate.raise_if_empty_str(val=exp_name, val_name='exp_name')

  # `ckpt_dir_path` validation.
  ckpt_dir_path = os.path.join(lmp.vars.EXP_PATH, exp_name)
  lmp.util.validate.raise_if_is_file(path=ckpt_dir_path)

  # Load the last checkpoint if `ckpt == -1`.
  if ckpt == -1:
    for ckpt_file_name in os.listdir(ckpt_dir_path):
      match = re.match(r'model-(\d+).pt', ckpt_file_name)
      if match is None:
        continue
      ckpt = max(int(match.group(1)), ckpt)

  # `ckpt_file_path` validation.
  ckpt_file_path = os.path.join(ckpt_dir_path, f'model-{ckpt}.pt')
  lmp.util.validate.raise_if_is_directory(path=ckpt_file_path)

  return torch.load(ckpt_file_path)


def list_ckpts(exp_name: str, first_ckpt: int, last_ckpt: int) -> List[int]:
  r"""List all pre-trained model checkpoints from ``first_ckpt`` to ``last_ckpt``.

  The last checkpoint is included.

  Parameters
  ----------
  exp_name: str
    Pre-trained language model experiment name.
  first_ckpt: int
    First checkpoint to include.
    Set to ``-1`` to include only the last checkpoint.
  last_ckpt: int
    Last checkpoint to include.
    Set to ``-1`` to include all checkpoints whose number is greater than ``first_ckpt``.

  Returns
  -------
  list[int]
    All available checkpoints of the experiment.
    Checkpoints are sorted in ascending order.
  """
  # `exp_name` validation.
  lmp.util.validate.raise_if_not_instance(val=exp_name, val_name='exp_name', val_type=str)
  lmp.util.validate.raise_if_empty_str(val=exp_name, val_name='exp_name')

  # `first_ckpt` validation.
  lmp.util.validate.raise_if_not_instance(val=first_ckpt, val_name='first_ckpt', val_type=int)
  lmp.util.validate.raise_if_wrong_ordered(vals=[-1, first_ckpt], val_names=['-1', 'first_ckpt'])

  # `last_ckpt` validation.
  lmp.util.validate.raise_if_not_instance(val=last_ckpt, val_name='last_ckpt', val_type=int)
  lmp.util.validate.raise_if_wrong_ordered(vals=[-1, last_ckpt], val_names=['-1', 'last_ckpt'])

  # `ckpt_dir_path` validation.
  ckpt_dir_path = os.path.join(lmp.vars.EXP_PATH, exp_name)
  lmp.util.validate.raise_if_is_file(path=ckpt_dir_path)

  ckpt_list = []
  for ckpt_file_name in os.listdir(ckpt_dir_path):
    match = re.match(r'model-(\d+).pt', ckpt_file_name)
    if match is None:
      continue
    ckpt_list.append(int(match.group(1)))

  if first_ckpt == -1:
    return [max(ckpt_list)]

  if last_ckpt == -1:
    last_ckpt = max(ckpt_list)

  return sorted(list(filter(lambda ckpt: first_ckpt <= ckpt <= last_ckpt, ckpt_list)))
