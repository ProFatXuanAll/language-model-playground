import lmp.dset
import lmp.model
import lmp.script.eval_dset_ppl
import lmp.script.train_model
import lmp.script.train_tknzr
import lmp.tknzr

lmp.script.train_tknzr.main(
    argv=[
        'whitespace', # type of tknzr, can check in lmp/tknzr
        '--dset_name',
        'demo', # type of dataset, can check in lpm/dset
        '--exp_name',
        'demo_tknzr', # can name by yourself
        '--is_uncased',
        '--max_vocab',
        '-1',
        '--min_count',
        '0',
        '--ver',
        'train'
    ]
)

for d_emb in [10, 100]:
    for d_hid in [10, 100]:
        for n_lyr in [1, 2, 3]:
            exp_name = f'demo-d_emb-{d_emb}-d_hid-{d_hid}-n_lyr-{n_lyr}' # can name by yourself
            lmp.script.train_model.main(
                argv=[
                    'Elman-Net', # type of model, can check in lpm/model
                    '--batch_size',
                    '32',
                    '--beta1',
                    '0.9',
                    '--beta2',
                    '0.999',
                    '--ckpt_step',
                    '500',
                    '--d_emb',
                    str(d_emb),
                    '--d_hid',
                    str(d_hid),
                    '--dset_name',
                    'demo', # type of dataset, can check in lpm/dset
                    '--eps',
                    '1e-8',
                    '--exp_name',
                    exp_name,
                    '--init_lower',
                    '-0.1',
                    '--init_upper',
                    '0.1',
                    '--label_smoothing',
                    '0.0',
                    '--log_step',
                    '100',
                    '--lr',
                    '1e-3',
                    '--max_norm',
                    '1',
                    '--max_seq_len',
                    '35',
                    '--n_lyr',
                    str(n_lyr),
                    '--p_emb',
                    '0.0',
                    '--p_hid',
                    '0.0',
                    '--seed',
                    '42',
                    '--stride',
                    '35',
                    '--tknzr_exp_name',
                    'demo_tknzr', # remember to follow the name above
                    '--total_step',
                    '40000',
                    '--ver',
                    'train',
                    '--warmup_step',
                    '10000',
                    '--weight_decay',
                    '0.0'
                ]
            )

            for ver in lmp.dset.DemoDset.vers: # remember to select the same dataset as above
                lmp.script.eval_dset_ppl.main(
                    argv=[
                        'demo',
                        '--batch_size',
                        '512',
                        '--exp_name',
                        exp_name,
                        '--first_ckpt',
                        '0',
                        '--last_ckpt',
                        '-1',
                        '--seed',
                        '42',
                        '--ver',
                        ver
                    ]
                )