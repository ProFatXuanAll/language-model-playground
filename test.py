import argparse

a = 'djjsada'
def tmp(aa):
    print(aa)

save_path = 'D:/Hsiu_Wen'
for i in range(100):
    if i %10 == 0:
        print(f'{save_path}/model{i}.ckpt')