#!/bin/bash
# Activate conda environment and run nvidia-smi
. /opt/conda/etc/profile.d/conda.sh
conda activate hpc
nvidia-smi

# Define paths
_path="/jmain02/home/J2AD006/jxb06/cxz46-jxb06/Block_ExperienceReplay/"
_log="/jmain02/home/J2AD006/jxb06/cxz46-jxb06/logging/Qbert"
_checkpoint="/jmain02/home/J2AD006/jxb06/cxz46-jxb06/checkpoints/Qbert"

# Change directory
cd $_path || exit

# Run Python script
python $_path/ddqn.py \
    -S $_path/settings/ddqn/Qbert.yml \
    -R $SLURM_JOB_ID \
    -L $_log \
    -C $_checkpoint
