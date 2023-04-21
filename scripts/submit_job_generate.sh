#!/bin/bash
#SBATCH --partition=ml
#SBATCH --job-name=cryonet-generate
#SBATCH --output=output-%j.txt --error=output-%j.txt
#SBATCH --nodes=1
#SBATCH --gpus=a100:1
#SBATCH --mem=256000

if [ $# -ne 4 ]; then
  echo "ERROR USAGE: $0 <config-path> <dataset-path> <dataset-name> <model pdb file>"
  echo You need to specify a train config.ini, a path for saving the dataset, a dataset name, and a PDB file
  exit 1
fi

singularity exec -B /sdf --nv /sdf/group/ml/CryoNet/singularity_images/animate_latest.sif \
            python src/experiment_scripts/generate_dataset.py -c "$1" --sim_starfile_dir "$2" --experiment_name "$3" --atomic_pdb "$4"
