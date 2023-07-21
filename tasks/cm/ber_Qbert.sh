#!/bin/bash
nvidia-smi
_path=/jmain02/home/J2AD006/jxb06/cxz46-jxb06/Block_ExperienceReplay/
cd $_path
python $_path/basic_dqn.py -S $_path/settings/basic_dqn/Qbert.yml -SBZ 8
