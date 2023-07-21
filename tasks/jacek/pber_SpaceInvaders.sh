#!/bin/bash
nvidia-smi
_path=/jmain02/home/J2AD006/jxb06/jxc15-jxb06/Block_ExperienceReplay/
cd $_path
python $_path/ddqn.py -S $_path/settings/ddqn/SpaceInvaders.yml -SBZ 4
