# import os

# for lmbda in [0.004,  0.0005]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
#     for cuda, scene in enumerate(['treehill', 'flowers', 'bonsai', ]): # ['bicycle', 'garden', 'stump', 'room', 'counter', 'kitchen', 'bonsai', 'flowers', 'treehill']
#         one_cmd = f'CUDA_VISIBLE_DEVICES={3} python train.py -s data/mipnerf360/{scene} --eval --lod 0 --voxel_size 0.001 --update_init_factor 16 --iterations 30_000 -m outputs/final_ours/mipnerf360/{scene}_{lmbda} --lmbda {lmbda} --target_ratio 0.2 --use_wandb'
#         os.system(one_cmd)

import os

# for lmbda in [1.1]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
#     for cuda, scene in enumerate(["bicycle"]):# enumerate(['bicycle', 'garden', 'stump', 'room', 'counter', 'kitchen', 'bonsai', 'flowers', 'treehill']):
#         one_cmd = f'CUDA_VISIBLE_DEVICES={0} python train.py -s data/mipnerf360/{scene} --eval --lod 0 --voxel_size 0.001 --update_init_factor 16 --iterations 30_000 -m outputs/final_ours_lmbda_rec/mipnerf360/{scene}_{lmbda} --lmbda_rec {lmbda} --target_ratio 0.2' # --use_wandb
#         os.system(one_cmd)

# import os

# for lmbda in [0.0005]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
#     for cuda, scene in enumerate(['bonsai']): # ['bicycle', 'garden', 'stump', 'room', 'counter', 'kitchen', 'bonsai', 'flowers', 'treehill']
#         one_cmd = f'CUDA_VISIBLE_DEVICES={2} python train.py -s data/mipnerf360/{scene} --eval --lod 0 --voxel_size 0.001 --update_init_factor 16 --iterations 30_000 -m outputs/final_ours/mipnerf360/{scene}_{lmbda} --lmbda {lmbda} --target_ratio 0.2 --use_wandb'
#         os.system(one_cmd)

for lmbda in [0.004,  0.0005]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
    for cuda, scene in enumerate(['bicycle', 'garden', 'stump', 'room', 'counter', 'kitchen', 'bonsai', 'flowers', 'treehill']):
        one_cmd = f'CUDA_VISIBLE_DEVICES={3} python test.py -s data/mipnerf360/{scene} --eval --lod 0 --voxel_size 0.001 -m outputs/final_ours/mipnerf360/{scene}_{lmbda} --target_ratio 0.2'
        os.system(one_cmd)
