# import os

# for lmbda in [0.004, 0.003, 0.002, 0.001, 0.0005]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
#     for cuda, scene in enumerate(['truck', 'train']):
#         one_cmd = f'CUDA_VISIBLE_DEVICES={0} python train.py -s data/tandt/{scene} --eval --lod 0 --voxel_size 0.01 --update_init_factor 16 --iterations 30_000 -m outputs/final_ours/tandt/{scene}_{lmbda} --lmbda {lmbda} --use_wandb'
#         os.system(one_cmd)




# import os

# for lmbda_rec in [2, 3]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
#     for cuda, scene in enumerate(['truck', 'train']):
#         one_cmd = f'CUDA_VISIBLE_DEVICES={1} python train.py -s data/tandt/{scene} --eval --lod 0 --voxel_size 0.01 --update_init_factor 16 --iterations 30_000 -m outputs/final_ours_lmbda_rec/tandt/{scene}_{lmbda_rec} --lmbda_rec {lmbda_rec} --use_wandb'
#         os.system(one_cmd)

# import os
# for lmbda in [0.0005]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
#     for cuda, scene in enumerate(['truck']):
#         one_cmd = f'CUDA_VISIBLE_DEVICES={3} python train.py -s data/tandt/{scene} --eval --lod 0 --voxel_size 0.01 --update_init_factor 16 --iterations 30_000 -m outputs/final_ours_multi_render/tandt/{scene}_{lmbda} --lmbda {lmbda} --use_wandb'
#         os.system(one_cmd)



import os

for lmbda in [0.004, 0.003, 0.002, 0.001, 0.0005]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
    for cuda, scene in enumerate(['truck', 'train']):
        one_cmd = f'CUDA_VISIBLE_DEVICES={2} python test.py -s data/tandt/{scene} --eval --lod 0 --voxel_size 0.01 --iterations 30_000 -m outputs/final_ours/tandt/{scene}_{lmbda}'
        os.system(one_cmd)
