import re
import torch

import lmp.dset
import lmp.infer
import lmp.model
import lmp.script
import lmp.tknzr
import lmp.util.model
import lmp.util.tknzr

device = torch.device('cuda')
tknzr = lmp.util.tknzr.load(exp_name='demo_tknzr')

for d_emb in [10, 100]:
  for d_hid in [10, 100]:
    for n_lyr in [1, 2, 3]:
      for ckpt in [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000]:
        for ver in lmp.dset.DemoDset.vers:
          dset = lmp.dset.DemoDset(ver=ver)
          exp_name = f'demo-d_emb-{d_emb}-d_hid-{d_hid}-n_lyr-{n_lyr}'
          model = lmp.util.model.load(exp_name=exp_name, ckpt=ckpt).to(device)
          infer = lmp.infer.Top1Infer(max_seq_len=35)

          correct = 0
          for spl in dset:
            match = re.match(r'If you add (\d+) to (\d+) you get (\d+) .', spl)
            input = f'If you add {match.group(1)} to {match.group(2)} you get '

            output = infer.gen(model=model, tknzr=tknzr, txt=input)

            if input + output == spl:
              correct += 1

          print(f'{exp_name}, ckpt: {ckpt}, ver: {ver}, acc: {correct / len(dset) * 100 :.2f}%')