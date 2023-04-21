# aNiMAte
aNiMate is a self-supervised reconstruction approach that eliminates post-hoc atomic model fitting into 3D maps by directly providing atomistic details of the structural spread of arbitrarily sized molecules captured by cryo-EM, through deformation of a given atomic model along its normal modes

`sbatch -t 1:00:00 scripts/submit_job_generate.sh configs/ak-atomic-primal-sim.ini sim-data`

`sbatch -t 1:00:00 scripts/submit_job.sh configs/ak-atomic-primal-sim.ini`