#!/bin/bash
# Activate conda environment and run nvidia-smi
. /opt/conda/etc/profile.d/conda.sh
conda activate ber
nvidia-smi

# Define paths
_path="/jmain02/home/J2AD006/jxb06/cxz46-jxb06/Block_ExperienceReplay/"
_log="/jmain02/home/J2AD006/jxb06/cxz46-jxb06/logging/Breakout"
_checkpoint="/jmain02/home/J2AD006/jxb06/cxz46-jxb06/checkpoints/Breakout"

# Change directory
cd $_path || exit

# Run Python script
python $_path/apex_dqn.py \
    -S $_path/settings/apex_ddqn/Breakout.yml \
    -R $SLURM_JOB_ID \
    -L $_log \
    -C $_checkpoint \
    -SBZ 16
