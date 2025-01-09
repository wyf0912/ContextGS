import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-l", "--lmbda", nargs="+", type=float, default=[0.004])
args = parser.parse_args()

for lmbda in args.lmbda:
    for cuda, scene in enumerate(['amsterdam', 'bilbao', 'hollywood', 'pompidou', 'quebec', 'rome']):
        one_cmd = f'CUDA_VISIBLE_DEVICES={args.gpu} python train.py -s data/bungeenerf/{scene} --eval --lod 30 --voxel_size 0 --update_init_factor 128 --iterations 30_000 -m outputs/retrain3_final_ours/bungeenerf/{scene}_{lmbda} --lmbda {lmbda} --level_scale 3 --target_ratio 0.2 --use_wandb' # --checkpoint_iterations 29999
        os.system(one_cmd)