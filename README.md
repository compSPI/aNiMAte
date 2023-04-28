# aNiMAte

![image](https://user-images.githubusercontent.com/696719/234671408-a180f593-bb26-4742-b11b-42f75b25be3a.png)

aNiMate is an unsupervised reconstruction approach that eliminates post-hoc atomic model fitting into 3D maps by directly providing atomistic details of the structural spread of arbitrarily sized molecules captured by cryo-EM, through deformation of a given atomic model along its normal modes.

The following documentation is specific to running the code on SLAC Shared Scientific
Data Facility ([SDF](https://sdf.slac.stanford.edu/public/doc/#/)). Dependencies are handled listed in [requirements.txt](requirements.txt) and are already incorporated in a Docker/Singularity image that's available on SDF at `/sdf/group/ml/CryoNet/singularity_images/animate_latest.sif` and on [Dockerhub](https://hub.docker.com/r/slaclab/animate). Any changes to [requirements.txt](requirements.txt) or [Dockerfile](Dockerfile) in this repo will lead to the docker image being rebuilt and pushed to Dockerhub, but the Singularity image will need to be manually pulled on SDF:
```
cd /sdf/group/ml/CryoNet/singularity_images
singularity pull -F docker://slaclab/animate:latest
```
## Getting Started
After cloning the code, everything should be run from within the upper directory of the code. SDF SLURM job submission are included under [scripts](scripts) and they rely on config files like the ones included under [configs](configs). The config files specify the arguments passed to the three modes the code can run in: 1) Generating datasets (Relion starfiles), 2) Training (from starfiles or simulation), and 3) Evaluation (using a checkpoint after/during training)


### Generating Datasets
In generation mode, the code generates a simulated cryo-EM dataset (particles `.mrcs` files and `.star` relion 3.1 starfile), starting from an atomic model PDB files. The simulation parameters are specified in a config file. An example simulation for Adenylate kinase is included under [configs/ak-atomic-primal-sim.ini](configs/ak-atomic-primal-sim.ini), and can be run on SDF as follows:
```
sbatch -t 1:00:00 scripts/submit_job_generate.sh configs/ak-atomic-primal-sim.ini sim-data
```
This will generate a dataset under the relative directory `sim-data`, but a fully resolved path can also be passed as the second argument to the SLURM script.

### Training
There are two submodes for training a model: 1) with simulated data that is generated on-the-fly using NMA (same dynamic model as the reconstructed model), and 2) with data read from relion 3.1 starfiles (like the one simulated above or experimental datasets). An example simulation model training can be run as follows:
```
sbatch -t 1:00:00 scripts/submit_job.sh configs/ak-atomic-primal-sim.ini
```
While the model is training, intermediate results (tensorboard logs, model checkpoints, etc...) are saved under `./logs` with each training run as a subdirectory named after the SLURM job ID (ex. `logs/8007570_0`). Training can be monitored on SDF by running a tensorboard instance pointing to the logs directory.
```
singularity exec -B /sdf /sdf/group/ml/CryoNet/singularity_images/animate_latest.sif tensorboard --logdir=logs/ --port=6007 &
```
#### Reproducing Paper Results
For large bio-molecules, like the ones in the aNiMAte paper, you'll most likely need to run on multiple GPUs. The atomic models and their corresponding precalculated NMA modes are included in this repo under [data](data). The datasets (starfiles) used in the paper are on already on SDF, so in order to submit a training run for the Spliceosome on the GTX 2080 GPUs, run something like:
```
sbatch --nodes 2 --gres gpu:geforce_rtx_2080_ti:8 --cpus-per-task=32 -t 24:00:00 scripts/submit_job.sh configs/splice-atomic-primal-relion.ini
```
This will run a training run on 16 GPUs with 4 threads per GPU for data IO. Each GPU outputs its own log under `logs/[SLURM_JOB_ID]_{GPU_INDEX}`. Similarily, a training run for the Ribosome can be run using [configs/ribosome-atomic-primal-relion.ini](configs/ribosome-atomic-primal-relion.ini)

### Evaluation/Inference
