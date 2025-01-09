# import os

# for lmbda in [0.004, 0.003, 0.002, 0.001, 0.0005]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
#     for cuda, scene in enumerate(['playroom', 'drjohnson']):
#         one_cmd = f'CUDA_VISIBLE_DEVICES={1} python train.py -s data/blending/{scene} --eval --lod 0 --voxel_size 0.005 --update_init_factor 16 --iterations 30_000 -m outputs/final_ours/blending/{scene}_{lmbda} --lmbda {lmbda} --use_wandb'
#         os.system(one_cmd)

# import os

# for lmbda in [2, 3]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
#     for cuda, scene in enumerate(['playroom', 'drjohnson']):
#         one_cmd = f'CUDA_VISIBLE_DEVICES={2} python train.py -s data/blending/{scene} --eval --lod 0 --voxel_size 0.005 --update_init_factor 16 --iterations 30_000 -m outputs/final_ours_lmbda_rec/blending/{scene}_{lmbda} --lmbda_rec {lmbda} --use_wandb'
#         os.system(one_cmd)


import os

for lmbda in [0.0005, 0.001]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
    for cuda, scene in enumerate(['playroom']):
        one_cmd = f'CUDA_VISIBLE_DEVICES={1} python train.py -s data/blending/{scene} --eval --lod 0 --voxel_size 0.005 --update_init_factor 16 --iterations 30_000 -m outputs/final_ours/blending/{scene}_{lmbda} --lmbda {lmbda} --use_wandb'
        os.system(one_cmd)
