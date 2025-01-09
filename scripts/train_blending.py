import os

for lmbda in [0.004, 0.0005, 0.003, 0.002, 0.001]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
    for cuda, scene in enumerate(['playroom', 'drjohnson']):
        one_cmd = f'CUDA_VISIBLE_DEVICES={1} python train.py -s data/blending/{scene} --eval --lod 0 --voxel_size 0.005 --update_init_factor 16 --iterations 30_000 -m outputs/retrain3_final_ours/blending_CE_1/{scene}_{lmbda} --lmbda {lmbda} --use_wandb'
        os.system(one_cmd)

