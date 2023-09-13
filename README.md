# aNiMAte

![image](https://user-images.githubusercontent.com/696719/234671408-a180f593-bb26-4742-b11b-42f75b25be3a.png)

aNiMate is an unsupervised reconstruction approach that eliminates post-hoc atomic model fitting into 3D maps by directly providing atomistic details of the structural spread of arbitrarily sized molecules captured by cryo-EM, through deformation of a given atomic model along its normal modes.

Dependencies are listed in [requirements.txt](requirements.txt) and are already incorporated in a Docker/Singularity image that's available on [Dockerhub](https://hub.docker.com/repository/docker/fpoitevi/animate). You can build your own image by editing the [Dockerfile recipe](Dockerfile) provided and running `docker build`. 

> **SLAC central install**
> 
> aNiMAte has been extensively tested at the SLAC Shared Scientific Data Facility ([SDF](https://sdf.slac.stanford.edu/public/doc/#/)) where the Singularity image is available at `/sdf/group/ml/CryoNet/singularity_images/animate_latest.sif`. To update it:
> ```
> cd /sdf/group/ml/CryoNet/singularity_images
> singularity pull -F docker://fpoitevi/animate:latest
> ```

## Getting Started
After cloning the code, everything should be run from within the upper directory of the code. SDF SLURM job submission are included under [scripts](scripts) and they rely on config files like the ones included under [configs](configs). The config files specify the arguments passed to the three modes the code can run in: 1) Generating datasets (Relion starfiles), 2) Training (from starfiles or simulation), and 3) Evaluation (using a checkpoint after/during training).


### 1- Generating Datasets
In generation mode, the code generates a simulated cryo-EM dataset (particles `.mrcs` files and `.star` relion 3.1 starfile), starting from an atomic model PDB files. The simulation parameters are specified in a config file. An example simulation for Adenylate kinase is included under [configs/ak-atomic-primal-sim.ini](configs/ak-atomic-primal-sim.ini), and can be run on SDF as follows:
```
sbatch -t 1:00:00 scripts/submit_job_generate.sh configs/ak-atomic-primal-sim.ini sim-data
```
This will generate a dataset under the relative directory `sim-data`, but a fully resolved path can also be passed as the second argument to the SLURM script.

### 2- Training
There are two submodes for training a model: 1) with simulated data that is generated on-the-fly using NMA (same dynamic model as the reconstructed model), and 2) with data read from relion 3.1 starfiles (like the one simulated above or experimental datasets). An example simulation model training can be run as follows:
```
sbatch -t 1:00:00 --gpus 1 scripts/submit_job.sh configs/ak-atomic-primal-sim.ini
```
While the model is training, intermediate results (tensorboard logs, model checkpoints, etc...) are saved under `./logs` with each training run as a subdirectory named after the SLURM job ID (ex. `logs/8007570_0`). Training can be monitored on SDF by running a tensorboard server pointing to the logs directory.
```
singularity exec -B /sdf /sdf/group/ml/CryoNet/singularity_images/animate_latest.sif tensorboard --logdir=logs/ --port=6007 &
```
#### Reproducing Paper Results
For large bio-molecules, like the ones in the aNiMAte paper, you'll most likely need to run on multiple GPUs. The atomic models and their corresponding precalculated NMA modes are included in this repo under [data](data). The datasets (starfiles) used in the paper are on already on SDF, so in order to submit a training run for the Spliceosome on the GTX 2080 GPUs, run something like:
```
sbatch --nodes 2 --gres gpu:geforce_rtx_2080_ti:8 --cpus-per-task 32 -t 24:00:00 scripts/submit_job.sh configs/splice-atomic-primal-relion.ini
```
This will run a training run on 16 GPUs with 4 threads per GPU for data IO. Each GPU outputs its own log under `logs/[SLURM_JOB_ID]_{GPU_INDEX}`. Similarily, a training run for the Ribosome can be run using [configs/ribosome-atomic-primal-relion.ini](configs/ribosome-atomic-primal-relion.ini)

### 3- Evaluation/Inference
After/During a training run, an evaluation run can be submitted with the provided [scripts/submit_job_eval.sh](scripts/submit_job_eval.sh) script. Example below. 
```
sbatch -t 02:00:00 -n 16 scripts/submit_job_eval.sh logs/8007570_0/config.ini logs/8007570_0/models/checkpoints/model_current.pth
```
This script takes two arguments, the first specifying the training config file, and the second pointing to a specific model checkpoint to load.
Usually, you'll need to provide the evaluation dataset/starfile, which can be the same as the training starfile. Add or update the `val_relion_star_file` config argument in the config INI file passed to `submit_job_eval.sh` first before running the above command. 

After evaluation is done, a new subdirectory will be created under logs with the format `logs/[SLURM_JOB_ID]` that contain the evaluation output and NMA plots. More visualizations can be generated from the provided Jupyter notebook [notebooks/inspect_predictions.ipynb](notebooks/inspect_predictions.ipynb). Furthermore, a notebook is provided to evaluate the "Delta" score described in the paper: [notebooks/evaluate_delta_scores.ipynb](notebooks/evaluate_delta_scores.ipynb). 

#### Accessing Paper Results

The actual runs/models for the paper with their corresponding job IDs are listed below. They can be accessed on SDF or downloaded from Google Drive mirroring `/sdf/group/ml/CryoNet/aNiMAte/publication` which can be found at [https://bit.ly/aNiMAte_data](https://bit.ly/aNiMAte_data). In that Google Drive directory, each subdirectory corresponds to one dataset which in turn contains 3 directories: configs/, data/ and logs/ which respectively contain the input config and data files as well as the curated output files.

| Description | SLURM Job ID | original SDF Path | curated SDF Path |
| ----------- | ----------- | ----------- | ----------- |
| Spliceosome, trained on half 1 | 7993005 | `/sdf/group/ml/CryoNet/cryonettorch/logs/7993005_0` | `/sdf/group/ml/CryoNet/aNiMAte/publication/empiar10180/logs/7993005_0` |
| spliceosome, evaluated on half 1 | | `/sdf/group/ml/CryoNet/cryonettorch/logs/7993005` | `/sdf/group/ml/CryoNet/aNiMAte/publication/empiar10180/logs/7993005` |
| Spliceosome, trained on half 2 | 7997982 | `/sdf/group/ml/CryoNet/cryonettorch/logs/7997982_0` | `/sdf/group/ml/CryoNet/aNiMAte/publication/empiar10180/logs/7997982_0` |
| Spliceosome, evaluated on half 2 | | `/sdf/group/ml/CryoNet/cryonettorch/logs/7997982` | `/sdf/group/ml/CryoNet/aNiMAte/publication/empiar10180/logs/7997982` |
| Spliceosome, trained on full data | 8008310 | `/sdf/group/ml/CryoNet/cryonettorch/logs/8008310_0` | `/sdf/group/ml/CryoNet/aNiMAte/publication/empiar10180/logs/8008310_0` |
| Spliceosome, evaluated on full data | | `/sdf/group/ml/CryoNet/cryonettorch/logs/8008310` | `/sdf/group/ml/CryoNet/aNiMAte/publication/empiar10180/logs/8008310` |
| Spliceosome (no-head), trained on half 1 | 8007570 | `/sdf/group/ml/CryoNet/cryonettorch/logs/8007570_0` | |
| Spliceosome (no-head), trained on half 2 | 7992273 | `/sdf/group/ml/CryoNet/cryonettorch/logs/7992273_0` | |
| Spliceosome (no-head), trained on full data | 8019309 | `/sdf/group/ml/CryoNet/cryonettorch/logs/8019309_0` | `/sdf/group/ml/CryoNet/aNiMAte/publication/empiar10180/logs/8019309_0`|
| Spliceosome (no-head), evaluated on full data | | `/sdf/group/ml/CryoNet/cryonettorch/logs/8019309` | `/sdf/group/ml/CryoNet/aNiMAte/publication/empiar10180/logs/8019309` |
| Ribosome, trained on half 1 | 10438442 | `/sdf/group/ml/CryoNet/aNiMAte/logs/10438442_0` | `/sdf/group/ml/CryoNet/aNiMAte/publication/empiar10028/logs/10438442_0` |
| Ribosome, evaluated on half 1 | | `/sdf/group/ml/CryoNet/aNiMAte/logs/10438442` | `/sdf/group/ml/CryoNet/aNiMAte/publication/empiar10028/logs/10438442` |
| Ribosome, trained on half 2 | 7971120 | `/sdf/group/ml/CryoNet/cryonettorch/logs/7971120_0` | `/sdf/group/ml/CryoNet/aNiMAte/publication/empiar10028/logs/7971120_0` |
| Ribosome, evaluated on half 2 | | `/sdf/group/ml/CryoNet/cryonettorch/logs/7971120` | `/sdf/group/ml/CryoNet/aNiMAte/publication/empiar10028/logs/7971120` |
| Ribosome, trained on full data | 8007515 | `/sdf/group/ml/CryoNet/cryonettorch/logs/8007515_0` | `/sdf/group/ml/CryoNet/aNiMAte/publication/empiar10028/logs/8007515_0` |
| Ribosome, evaluated on full data | | `/sdf/group/ml/CryoNet/cryonettorch/logs/8007515` | `/sdf/group/ml/CryoNet/aNiMAte/publication/empiar10028/logs/8007515` |
| Ribosome (no-ES12), trained on half 1 | 8333192 | `/sdf/group/ml/CryoNet/cryonettorch/logs/8333192_0` | |
| Ribosome (no-ES12), trained on half 2 | 8333194 | `/sdf/group/ml/CryoNet/cryonettorch/logs/8333194_0` | |
| Ribosome (no-ES12), trained on full data | 8209068 | `/sdf/group/ml/CryoNet/cryonettorch/logs/8209068_0` | `/sdf/group/ml/CryoNet/aNiMAte/publication/empiar10028/logs/8209068_0` |
| Ribosome (no-ES12), evaluated on full data | | `/sdf/group/ml/CryoNet/cryonettorch/logs/8209068` | `/sdf/group/ml/CryoNet/aNiMAte/publication/empiar10028/logs/8209068` |
