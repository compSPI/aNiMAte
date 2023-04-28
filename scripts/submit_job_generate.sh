#!/bin/bash
#SBATCH --partition=ml
#SBATCH --job-name=aNiMAte-generate
#SBATCH --output=output-%j.txt --error=output-%j.txt
#SBATCH --nodes=1
#SBATCH --gpus=a100:1
#SBATCH --mem=256000

if [ $# -ne 2 ]; then
  echo "ERROR USAGE: $0 <config-path> <out-path>"
  echo "You need to specify a train config.ini and a path for saving the dataset starfile and particle images"
  exit 1
fi

singularity exec -B /sdf --nv /sdf/group/ml/CryoNet/singularity_images/animate_latest.sif \
            python src/experiment_scripts/generate_dataset.py -c "$1" --sim_starfile_dir "$2"
