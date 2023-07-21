#!/bin/bash
nvidia-smi
_path=/jmain02/home/J2AD006/jxb06/cxz46-jxb06/Block_ExperienceReplay/
cd $_path
python $_path/ddqn.py -S $_path/settings/ddqn/Qbert.yml -SBZ 8
