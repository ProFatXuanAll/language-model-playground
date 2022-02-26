"""Test perplexity calculation result.

Test target:
- :py:meth:`lmp.script.ddp_eval_dset_ppl.main`.
"""

import multiprocessing as mp
import os
from typing import List

import lmp.script.ddp_eval_dset_ppl
from lmp.dset import WikiText2Dset


def test_ppl_output(
  batch_size: int,
  ckpts: List[int],
  eval_log_dir_path: str,
  host_name: str,
  host_port: int,
  is_dset_in_memory: bool,
  model_exp_name: str,
  n_worker: int,
  seed: int,
  world_size: int,
) -> None:
  """Must correctly output perplexity."""
  # This is need since one cannot re-initialize CUDA in forked subprocess.
  ctx = mp.get_context('spawn')

  argv = [
    WikiText2Dset.dset_name,
    '--batch_size',
    str(batch_size),
    '--exp_name',
    model_exp_name,
    '--first_ckpt',
    str(min(ckpts)),
    '--host_name',
    host_name,
    '--host_port',
    str(host_port),
    '--last_ckpt',
    str(max(ckpts)),
    '--seed',
    str(seed),
    '--ver',
    'valid',
    '--world_size',
    str(world_size),
  ]

  if is_dset_in_memory:
    argv.append('--is_dset_in_memory')

  if world_size == 1:
    argv.append('--rank')
    argv.append('0')
    argv.append('--local_rank')
    argv.append('0')
    lmp.script.ddp_eval_dset_ppl.main(argv=argv)
  else:
    pool = []
    for rank in range(world_size):
      p = ctx.Process(
        target=lmp.script.ddp_eval_dset_ppl.main,
        args=(argv + ['--rank', str(rank), '--local_rank', str(rank)],)
      )
      p.start()
      pool.append(p)

    for p in pool:
      p.join()

    assert all(map(lambda p: p.exitcode == 0, pool))

    for p in pool:
      p.close()

  assert os.path.exists(eval_log_dir_path)
