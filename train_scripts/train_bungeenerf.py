# import os

# for cuda, scene in enumerate(['amsterdam', 'bilbao', 'hollywood', 'pompidou', 'quebec', 'rome']):
#     for lmbda in [0.004, 0.0005, 0.003, 0.002, 0.001]: # ,
#         one_cmd = f'CUDA_VISIBLE_DEVICES={0} python train.py -s data/bungeenerf/{scene} --eval --lod 30 --voxel_size 0 --update_init_factor 128 --iterations 30_000 -m outputs/final_ours/bungeenerf/{scene}_{lmbda} --lmbda {lmbda} --level_scale 3 --target_ratio 0.2 --use_wandb'
#         print(one_cmd)
#         os.system(one_cmd)


# for cuda, scene in enumerate(['amsterdam', 'bilbao', 'hollywood', 'pompidou', 'quebec', 'rome']):
#     for lmbda_rec in [2, 3]: # ,
#         one_cmd = f'CUDA_VISIBLE_DEVICES={0} python train.py -s data/bungeenerf/{scene} --eval --lod 30 --voxel_size 0 --update_init_factor 128 --iterations 30_000 -m outputs/final_ours_lmbda_rec/bungeenerf/{scene}_{lmbda_rec} --lmbda_rec {lmbda_rec} --level_scale 3 --target_ratio 0.2 --use_wandb'
#         print(one_cmd)
#         os.system(one_cmd)

# for cuda, scene in enumerate(['rome']):
#     for lmbda_rec in [1]: # ,
#         one_cmd = f'CUDA_VISIBLE_DEVICES={1} python train.py -s data/bungeenerf/{scene} --eval --lod 30 --voxel_size 0 --update_init_factor 128 --iterations 30_000 -m outputs/final_ours_lmbda_rec/bungeenerf/{scene}_{lmbda_rec}_count_time_A5000 --lmbda_rec {lmbda_rec} --level_scale 3 --target_ratio 0.2 --use_wandb'
#         print(one_cmd)
#         os.system(one_cmd)

import os
for cuda, scene in enumerate(['amsterdam', 'bilbao', 'hollywood', 'pompidou', 'quebec', 'rome']):
    for lmbda in [0.004, 0.0005, 0.003, 0.002, 0.001]: # ,
        one_cmd = f'CUDA_VISIBLE_DEVICES={0} python test.py -s data/bungeenerf/{scene} --eval --lod 30 --voxel_size 4.399655153974891e-05 -m outputs/final_ours/bungeenerf/{scene}_{lmbda} --level_scale 3 --target_ratio 0.2'
        print(one_cmd)
        os.system(one_cmd)
