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
```
sbatch -t 1:00:00 scripts/submit_job.sh configs/ak-atomic-primal-sim.ini
```
