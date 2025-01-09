import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--lmbda", nargs="+", type=float, default=[0.004])
args = parser.parse_args()


for lmbda in args.lmbda:
    for cuda, scene in enumerate(['bicycle']): # 
        one_cmd = f'CUDA_VISIBLE_DEVICES={args.gpu} python train.py -s data/mipnerf360/{scene} --eval --lod 0 --voxel_size 0.001 --update_init_factor 16 --iterations 30_000 -m outputs/retrain3_final_ours/mipnerf360/{scene}_{lmbda} --lmbda {lmbda} --target_ratio 0.2 --use_wandb' #  --checkpoint_iterations 29999
        os.system(one_cmd)
