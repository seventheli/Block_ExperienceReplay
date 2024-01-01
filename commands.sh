sbatch --mem=90G \
       --gres=gpu:1 \
       --cpus-per-task=6 \
       --time=4-10:00:00 \
       -p small \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/Block_ExperienceReplay/tasks/dpber_LavaCrossingS9N3.sh