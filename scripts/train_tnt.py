import os

for lmbda in [0.0005, 0.003, 0.002, 0.001]:  # 0.004,  Optionally, you can try:
    for idx, scene in enumerate(['truck', 'train']):
        one_cmd = f'CUDA_VISIBLE_DEVICES={3} python train.py -s data/tandt/{scene} --eval --lod 0 --voxel_size 0.01 --update_init_factor 16 --iterations 30_000 -m outputs/retrain2_final_ours/tandt/{scene}_{lmbda} --lmbda {lmbda} --use_wandb' # --checkpoint_iterations 10000 29990
        os.system(one_cmd)
