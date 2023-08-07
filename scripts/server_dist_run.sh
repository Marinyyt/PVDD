#!/bin/bash

# create sbatch_run.sh

PORT=$((12000 + $RANDOM % 20000))

PARTITION=${PARTITION:-pixel} # default partition is pixel

if [ "$#" -ne 4 ]; then
    echo "########################################################"
    echo " Usage: ./server_run.sh [jobname] [gpu_num per-node] [cpu_num] [total gpu num] [other]."
    echo "########################################################"
    exit 1
fi

jobname=$1       #your job script name
# script_name=$5
path=$PWD
if [ ! -d "logs/${jobname}" ]; then
  mkdir -p logs/${jobname}
fi


sbatch_file=sbatch_run.sh
touch ${sbatch_file}
cat > ${sbatch_file} <<'endmsg'
#!/bin/bash
#MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0
endmsg

echo "#SBATCH -p ${PARTITION}" >> ${sbatch_file}
echo "#SBATCH --gres=gpu:$2" >> ${sbatch_file} # gpu nums of single node
echo "#SBATCH --ntasks-per-node=$2" >> ${sbatch_file} # max num of processing on each node, 1p:1GPU
echo "#SBATCH --cpus-per-task=$3" >> ${sbatch_file}
echo "#SBATCH --ntasks=$4" >> ${sbatch_file} # num of tasks, total GPUs
echo "#SBATCH --job-name=${jobname}" >> ${sbatch_file}
echo "#SBATCH -o ./logs/${jobname}/%j.txt" >> ${sbatch_file}  # log path
echo "srun --mpi=pmi2 --kill-on-bad-exit=1 python train.py --config ./configs/${jobname}.yaml --num_gpus $2 --save_path ./logs/${jobname}/ --distributed_training --port ${PORT}" >> ${sbatch_file}
echo "echo \"Submit the ${jobname} job by run \'sbatch\'\" " >>  ${sbatch_file}

cat sbatch_run.sh
echo "Submit the ${jobname} job by run 'sbatch ${sbatch_file}'"
sbatch sbatch_run.sh
