#!/bin/bash
module load python/anaconda3
source $condaDotFile
source activate gymnasium

nvidia-smi

which python

# Define paths
_path="/jmain02/home/J2AD006/jxb06/cxz46-jxb06/Block_ExperienceReplay/"
_log="/jmain02/home/J2AD006/jxb06/cxz46-jxb06/logging/minigrid/LavaCrossingS9N3/"
_checkpoint="/jmain02/home/J2AD006/jxb06/cxz46-jxb06/checkpoints/minigrid/LavaCrossingS9N3/"

# Change directory
cd $_path || exit

# Run Python script
python $_path/apex_dpber.py \
    -R $SLURM_JOB_ID \
    -S $_path/settings/apex.yml \
    -L $_log \
    -C $_checkpoint \
    -E LavaCrossingS9N3