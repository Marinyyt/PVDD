#!/bin/bash

# create sbatch_run.sh

PARTITION=${PARTITION:-pixel} # default partition is pixel

if [ "$#" -ne 3 ]; then
    echo "########################################################"
    echo " Usage: ./server_run.sh [jobname] [gpu_num] [cpu_num] [other]."
    echo "########################################################"
    exit 1
fi

jobname=$1       #your job script name
# script_name=$5
path=$PWD
if [ ! -d "/mnt/lustrenew/share_data/yuyitong/test_logs/${jobname}" ]; then
  mkdir -p /mnt/lustrenew/share_data/yuyitong/test_logs/${jobname}
fi


sbatch_file=sbatch_run.sh
touch ${sbatch_file}
cat > ${sbatch_file} <<'endmsg'
#!/bin/bash
#MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0
endmsg

echo "#SBATCH -p ${PARTITION}" >> ${sbatch_file}
echo "#SBATCH --gres=gpu:$2" >> ${sbatch_file}
echo "#SBATCH -N 1" >> ${sbatch_file} # num of task
echo "#SBATCH --cpus-per-task=$3" >> ${sbatch_file}
# echo "#SBATCH --nodelist=$4" >> ${sbatch_file}
echo "#SBATCH --job-name=${jobname}" >> ${sbatch_file}
echo "#SBATCH -o /mnt/lustrenew/share_data/yuyitong/test_logs/${jobname}/%j.txt" >> ${sbatch_file}  # log path
echo "srun --mpi=pmi2 --kill-on-bad-exit=1 python test_video_poc_window_server.py --model_file /mnt/lustrenew/share_data/yuyitong/logs/PVDD_rvidenet_charbo_bs1_aug_davis_model/model/g_net_100000.pth --save_path /mnt/lustrenew/share_data/yuyitong/test_logs/${jobname}/100000/ --test_path /mnt/lustrenew/share_data/yuyitong/data/PVDD/test/pocnew/ --num_frame 3" >> ${sbatch_file}

echo "echo \"Submit the ${jobname} job by run \'sbatch\'\" " >>  ${sbatch_file}

cat sbatch_run.sh
echo "Submit the ${jobname} job by run 'sbatch ${sbatch_file}'"
sbatch sbatch_run.sh
