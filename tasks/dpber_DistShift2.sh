#!/bin/bash
module load python/anaconda3
source $condaDotFile
source activate gymnasium

nvidia-smi

which python

# Define paths
_path="/home/seventheli//Block_ExperienceReplay/"
_log="/home/seventheli//logging/minigrid/DistShift2/"
_checkpoint="/home/seventheli//checkpoints/minigrid/DistShift2/"

# Change directory
cd $_path || exit

# Run Python script
python $_path/apex_dpber.py \
    -R $SLURM_JOB_ID \
    -S $_path/settings/apex.yml \
    -L $_log \
    -C $_checkpoint \
    -E DistShift2


