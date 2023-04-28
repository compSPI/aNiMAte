#!/bin/bash
#SBATCH --partition=ml
#SBATCH --job-name=aNiMAte-train
#SBATCH --output=output-%j.txt --error=output-%j.txt
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128GB

if [[ $1 == "" ]]; then
    echo You need to specify a train config INI file
    exit 1
fi

# Configuration
nproc_per_node=`echo $CUDA_VISIBLE_DEVICES | awk -v FS="," '{print NF}'`
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )
master_node=${nodes_array[0]}
master_addr=$(srun --nodes=1 --ntasks=1 -w $master_node hostname --ip-address)
worker_num=$(($SLURM_JOB_NUM_NODES))

SIF_FILE=/sdf/group/ml/CryoNet/singularity_images/animate_latest.sif

#export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^docker0,lo
export CUDA_LAUNCH_BLOCKING=1
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Loop over nodes and submit training tasks
for ((  node_rank=0; node_rank<$worker_num; node_rank++ ))
do
  node=${nodes_array[$node_rank]}
  srun -N 1 -n 1 -w $node singularity exec -B /sdf \
            --nv ${SIF_FILE} \
            python -m torch.distributed.run \
            --nproc_per_node=$nproc_per_node --nnodes=$SLURM_JOB_NUM_NODES \
            --node_rank=$node_rank --master_addr=$master_addr \
            src/experiment_scripts/main.py \
            --experiment_name ${SLURM_JOBID} \
            --train_num_workers $((SLURM_CPUS_PER_TASK / nproc_per_node)) \
            --config "$1" &
  pids[${node_rank}]=$!
done

# Wait for completion
for pid in ${pids[*]}; do
    wait $pid
done
