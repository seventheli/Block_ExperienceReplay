#!/bin/bash
. /opt/conda/etc/profile.d/conda.sh
conda activate hpc
nvidia-smi
_path=/jmain02/home/J2AD006/jxb06/cxz46-jxb06/Block_ExperienceReplay/
cd $_path || exit
python $_path/apex_dqn.py -S $_path/settings/apex_ddqn/BeamRider.yml -SBZ 32 -R $SLURM_JOB_ID
