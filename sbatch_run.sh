#!/bin/bash
#MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0
#SBATCH -p Pixel
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=PVDD_pvdd0815_02_charbo_bs1_pvdd_model
#SBATCH -o /mnt/lustrenew/share_data/yuyitong/logs/PVDD_pvdd0815_02_charbo_bs1_pvdd_model/%j.txt
srun --mpi=pmi2 --kill-on-bad-exit=1 python train.py --config ./configs/PVDD_pvdd0815_02_charbo_bs1_pvdd_model.yaml --num_gpus 1 --save_path /mnt/lustrenew/share_data/yuyitong/logs/PVDD_pvdd0815_02_charbo_bs1_pvdd_model/ 
echo "Submit the PVDD_pvdd0815_02_charbo_bs1_pvdd_model job by run \'sbatch\'" 
