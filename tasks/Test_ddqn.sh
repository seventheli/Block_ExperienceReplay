#!/bin/bash
. /opt/conda/etc/profile.d/conda.sh
conda activate hpc
nvidia-smi
_path=/jmain02/home/J2AD006/jxb06/cxz46-jxb06/Block_ExperienceReplay/
cd $_path || exit
python $_path/ddqn.py -S $_path/settings/ddqn/BeamRider.yml -SBZ 8 -R $SLURM_JOB_ID
