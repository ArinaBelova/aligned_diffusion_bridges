#!/bin/bash

#SBATCH --job-name=train-protein-conf

#SBATCH --mail-type=ALL

#SBATCH --mail-user=<arina.belova@hhi.fraunhofer.de>

#SBATCH --output=output_logs/%j_%x_${1}_${2}.out

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=16

#SBATCH --gpus=1

#SBATCH --mem=32G

#####################################################################################

K="$1"
H="$2"

# meta-data
DATE=`date`
# set output directories
OUTPUTFOLDER="$SLURM_JOB_NAME-$SLURM_JOB_ID"
OUTPUTPATH_JOB="/opt/output"
SUBMIT_DIR=`pwd`
OUTPUTPATH_LOCAL="$SUBMIT_DIR/runs/$OUTPUTFOLDER"
# create temporary output directory
#source "/etc/slurm/local_job_dir.sh"
export LOCAL_JOB_DIR=/data/local/jobs/${SLURM_JOB_ID}
mkdir -p "${LOCAL_JOB_DIR}/job_results"
#export APPTAINER_BINDPATH="${APPTAINER_BINDPATH},${LOCAL_JOB_DIR}"
#cp -r ${SLURM_SUBMIT_DIR}/cache_datasets ${LOCAL_JOB_DIR}

# Launch the apptainer image with --nv for nvidia support. Two bind mounts are used:
# - One for the ImageNet dataset and
# - One for the results (e.g. checkpoint data that you may store in $LOCAL_JOB_DIR on the node
# For debugging disable wandb
# apptainer exec --nv --bind ${LOCAL_JOB_DIR} \
# ./Reflected-Diffusion/cluster/reflected.sif \
# wandb enabled; 

#CUDA_VISIBLE_DEVICES=1,0

# To download the data
# apptainer exec --nv --bind ${LOCAL_JOB_DIR} \
# ./aligned_diffusion_bridges/cluster/sbalign.sif \
# wget --no-check-certificate -P /home/fe/belova/projects/bridges/aligned_diffusion_bridges/sbalign/data/processed "https://zenodo.org/records/8066711/files/d3pm_processed.tar.gz"

# apptainer exec --nv --bind ${LOCAL_JOB_DIR} \
# ./aligned_diffusion_bridges/cluster/sbalign.sif \
# tar xvzf /home/fe/belova/projects/bridges/aligned_diffusion_bridges/sbalign/data/processed/d3pm_processed.tar.gz -C /home/fe/belova/projects/bridges/aligned_diffusion_bridges/sbalign/data/processed/

# apptainer exec --nv --bind ${LOCAL_JOB_DIR} \
# ./aligned_diffusion_bridges/cluster/sbalign.sif \
# wget --no-check-certificate -P /home/fe/belova/projects/bridges/aligned_diffusion_bridges/sbalign/data/raw "https://zenodo.org/records/8066711/files/d3pm_raw.tar.gz"

# apptainer exec --nv --bind ${LOCAL_JOB_DIR} \
# ./aligned_diffusion_bridges/cluster/sbalign.sif \
# tar xvzf /home/fe/belova/projects/bridges/aligned_diffusion_bridges/sbalign/data/raw/d3pm_raw.tar.gz -C /home/fe/belova/projects/bridges/aligned_diffusion_bridges/sbalign/data/raw/

# apptainer exec --nv --bind ${LOCAL_JOB_DIR} \
# ./aligned_diffusion_bridges/cluster/sbalign.sif \
# python ${SLURM_SUBMIT_DIR}/aligned_diffusion_bridges/setup.py develop

# for wandb certificates to work:
export SSL_CERT_FILE=/home/fe/belova/projects/bridges/cacert.pem

# Before teh train don't forget to change the run_name argument in reproducibility/conf/config_train 
# train
apptainer exec --nv --bind ${LOCAL_JOB_DIR} \
./aligned_diffusion_bridges/cluster/sbalign.sif \
bash -c "
echo 'TRAINING STARTED WITH K=$K, H=$H'
python ${SLURM_SUBMIT_DIR}/aligned_diffusion_bridges/scripts/conf/train.py --config ${SLURM_SUBMIT_DIR}/aligned_diffusion_bridges/reproducibility/conf/config_train.yml --K ${K} --H ${H} --jobid ${SLURM_JOB_ID}
echo 'EVALUATION STARTED'
python ${SLURM_SUBMIT_DIR}/aligned_diffusion_bridges/scripts/conf/evaluate.py --data_dir ${SLURM_SUBMIT_DIR}/aligned_diffusion_bridges/sbalign/data --log_dir logs --run_name CONF_TP_MODEL-${SLURM_JOB_ID}-K-${K}-H-${H} \
    --model_name best_model.pt --method sbalign --inference_steps 100 --n_samples 10
"

# echo 'CHANGING CONFIG FILES WITH K=$K, H=$H'
# sed -i 's/^K: .*/K: $K/' ${SLURM_SUBMIT_DIR}/aligned_diffusion_bridges/reproducibility/conf/config_train.yml;
# sleep 1;
# sed -i 's/^H: .*/H: $H/' ${SLURM_SUBMIT_DIR}/aligned_diffusion_bridges/reproducibility/conf/config_train.yml;
# sleep 1;

# echo "EVALUATION STARTED"
# apptainer exec --nv --bind ${LOCAL_JOB_DIR} \
# ./aligned_diffusion_bridges/cluster/sbalign.sif \
# python ${SLURM_SUBMIT_DIR}/aligned_diffusion_bridges/scripts/conf/evaluate.py --data_dir ${SLURM_SUBMIT_DIR}/aligned_diffusion_bridges/sbalign/data --log_dir logs --run_name CONF_TP_MODEL-${SLURM_JOB_ID}-K-${K}-H-${H} \
#     --model_name best_model.pt --method sbalign --inference_steps 100 --n_samples 10

#wandb login "7d001f095c395e6c0fc4c23d85c2e9831b089ea7" \

# copying results from local
mkdir -p $OUTPUTPATH_LOCAL
cp -r ${LOCAL_JOB_DIR}/job_results/* $OUTPUTPATH_LOCAL
rm -r ${LOCAL_JOB_DIR}/job_results
# also copy output
cp "${SUBMIT_DIR}/runs/${SLURM_JOB_ID}_${SLURM_JOB_NAME}.out" "${SUBMIT_DIR}/runs/${SLURM_JOB_ID}"


# information about the outputs of the script
echo "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾"
echo " CONTENTS                 PATH                                                  "
echo "――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――"
echo " job_results              $OUTPUTPATH_LOCAL"
echo " .out file                ${SUBMIT_DIR}/runs/${SLURM_JOB_ID}"
echo "________________________________________________________________________________"