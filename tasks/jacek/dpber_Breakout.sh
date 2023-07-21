#!/bin/bash
nvidia-smi
_path=/jmain02/home/J2AD006/jxb06/jxc15-jxb06/Block_ExperienceReplay/
cd $_path
python $_path/apex_dqn.py -S $_path/settings/apex_ddqn/Breakout.yml -SBZ 32
