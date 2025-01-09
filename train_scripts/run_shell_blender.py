import os

for lmbda in [0.004, 0.0005, 0.003, 0.002, 0.001]:  # Optionally, you can try: 
    for cuda, scene in enumerate(['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']):
        one_cmd = f'CUDA_VISIBLE_DEVICES={1} python train.py -s data/nerf_synthetic/{scene} --eval --lod 0 --voxel_size 0.001 --update_init_factor 4 --iterations 30_000 -m outputs/final_ours/nerf_synthetic/{scene}_{lmbda} --lmbda {lmbda} --use_wandb'
        os.system(one_cmd)
