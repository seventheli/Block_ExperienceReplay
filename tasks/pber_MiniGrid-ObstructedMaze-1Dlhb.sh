#!/bin/bash
module load python/anaconda3
source $condaDotFile
source activate gymnasium

nvidia-smi

which python

# Define paths
_path="/jmain02/home/J2AD006/jxb06/cxz46-jxb06/Block_ExperienceReplay/"
_log="/jmain02/home/J2AD006/jxb06/cxz46-jxb06/logging/minigrid/MiniGrid-ObstructedMaze-1Dlhb/"
_checkpoint="/jmain02/home/J2AD006/jxb06/cxz46-jxb06/checkpoints/minigrid/MiniGrid-ObstructedMaze-1Dlhb/"

# Change directory
cd $_path || exit

# Run Python script
python $_path/ddqn_pber.py \
    -R $SLURM_JOB_ID \
    -S $_path/settings/ddqn.yml \
    -L $_log \
    -C $_checkpoint \
    -E MiniGrid-ObstructedMaze-1Dlhb
