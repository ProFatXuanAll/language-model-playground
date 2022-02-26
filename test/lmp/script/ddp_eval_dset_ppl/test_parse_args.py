"""Test parsing arguments.

Test target:
- :py:meth:`lmp.script.ddp_eval_dset_ppl.parse_args`.
"""

import lmp.script.ddp_eval_dset_ppl
from lmp.dset import DSET_OPTS


def test_parse_results(
  batch_size: int,
  is_dset_in_memory: bool,
  exp_name: str,
  host_name: str,
  host_port: int,
  n_worker: int,
  seed: int,
  world_size: int,
) -> None:
  """Must correctly parse all arguments."""
  rank = 0
  local_rank = 0

  for dset_name, dset_type in DSET_OPTS.items():
    for ver in dset_type.vers:
      for first_ckpt in [-1, 0, 1]:
        for last_ckpt in [-1, 0, 1]:
          argv = [
            dset_name,
            '--batch_size',
            str(batch_size),
            '--exp_name',
            exp_name,
            '--first_ckpt',
            str(first_ckpt),
            '--last_ckpt',
            str(last_ckpt),
            '--local_rank',
            str(local_rank),
            '--host_name',
            host_name,
            '--host_port',
            str(host_port),
            '--n_worker',
            str(n_worker),
            '--rank',
            str(rank),
            '--seed',
            str(seed),
            '--ver',
            ver,
            '--world_size',
            str(world_size),
          ]

          if is_dset_in_memory:
            argv.append('--is_dset_in_memory')

          args = lmp.script.ddp_eval_dset_ppl.parse_args(argv=argv)

          assert args.batch_size == batch_size
          assert args.dset_name == dset_name
          assert args.exp_name == exp_name
          assert args.first_ckpt == first_ckpt
          assert args.host_name == host_name
          assert args.host_port == host_port
          assert args.is_dset_in_memory == is_dset_in_memory
          assert args.last_ckpt == last_ckpt
          assert args.local_rank == local_rank
          assert args.n_worker == n_worker
          assert args.rank == rank
          assert args.seed == seed
          assert args.ver == ver
          assert args.world_size == world_size
