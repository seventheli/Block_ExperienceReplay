sbatch --mem=110G \
       --gres=gpu:1 \
       --cpus-per-task=15 \
       --time=4-10:00:00 \
       -p small \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/singbatch \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/image_conda.sif \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/Block_ExperienceReplay/tasks/Test_dqn.sh

sbatch --mem=110G \
       --gres=gpu:1 \
       --cpus-per-task=15 \
       --time=4-10:00:00 \
       -p small \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/singbatch \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/image_conda.sif \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/Block_ExperienceReplay/tasks/Test_apex.sh

sbatch --mem=110G \
       --gres=gpu:1 \
       --cpus-per-task=15 \
       --time=4-10:00:00 \
       -p small \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/singbatch \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/image_conda.sif \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/Block_ExperienceReplay/tasks/Test_ddqn.sh


sbatch --mem=110G \
       --gres=gpu:1 \
       --cpus-per-task=15 \
       --time=4-10:00:00 \
       -p small \
       /jmain02/home/J2AD006/jxb06/rxs17-jxb06/singbatch \
       /jmain02/home/J2AD006/jxb06/rxs17-jxb06/image_conda.sif \
       /jmain02/home/J2AD006/jxb06/rxs17-jxb06/Block_ExperienceReplay/tasks/cm/sr/pber_Qbert_4.sh