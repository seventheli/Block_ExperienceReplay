sbatch --mem=100G \
       --gres=gpu:1 \
       --cpus-per-task=15 \
       --time=4-10:00:00 \
       -p small \
       /jmain02/home/J2AD006/jxb06/jxc15-jxb06/Block_ExperienceReplay/tasks/Test_apex.sh

sbatch --mem=100G \
       --gres=gpu:1 \
       --cpus-per-task=15 \
       --time=4-10:00:00 \
       -p small \
       /jmain02/home/J2AD006/jxb06/jxc15-jxb06/singbatch \
       /jmain02/home/J2AD006/jxb06/jxc15-jxb06/image_conda.sif \
       /jmain02/home/J2AD006/jxb06/jxc15-jxb06/Block_ExperienceReplay/tasks/cm/jxc15/pber_Breakout_4.sh

sbatch --mem=100G \
       --gres=gpu:1 \
       --cpus-per-task=15 \
       --time=4-10:00:00 \
       -p small \
       /jmain02/home/J2AD006/jxb06/jxc15-jxb06/singbatch \
       /jmain02/home/J2AD006/jxb06/jxc15-jxb06/image_conda.sif \
       /jmain02/home/J2AD006/jxb06/jxc15-jxb06/Block_ExperienceReplay/tasks/cm/jxc15/pber_Qbert_4.sh

sbatch --mem=100G \
       --gres=gpu:1 \
       --cpus-per-task=15 \
       --time=4-10:00:00 \
       -p small \
       /jmain02/home/J2AD006/jxb06/jxc15-jxb06/singbatch \
       /jmain02/home/J2AD006/jxb06/jxc15-jxb06/image_conda.sif \
       /jmain02/home/J2AD006/jxb06/jxc15-jxb06/Block_ExperienceReplay/tasks/cm/jxc15/ber_Breakout_4.sh

sbatch --mem=100G \
       --gres=gpu:1 \
       --cpus-per-task=15 \
       --time=4-10:00:00 \
       -p small \
       /jmain02/home/J2AD006/jxb06/jxc15-jxb06/singbatch \
       /jmain02/home/J2AD006/jxb06/jxc15-jxb06/image_conda.sif \
       /jmain02/home/J2AD006/jxb06/jxc15-jxb06/Block_ExperienceReplay/tasks/cm/jxc15/pber_BeamRider_4.sh

sbatch --mem=100G \
       --gres=gpu:1 \
       --cpus-per-task=15 \
       --time=4-10:00:00 \
       -p small \
       /jmain02/home/J2AD006/jxb06/jxc15-jxb06/singbatch \
       /jmain02/home/J2AD006/jxb06/jxc15-jxb06/image_conda.sif \
       /jmain02/home/J2AD006/jxb06/jxc15-jxb06/Block_ExperienceReplay/tasks/cm/jxc15/ber_SpaceInvaders_4.sh

sbatch --mem=100G \
       --gres=gpu:1 \
       --cpus-per-task=15 \
       --time=4-10:00:00 \
       -p small \
       /jmain02/home/J2AD006/jxb06/jxc15-jxb06/singbatch \
       /jmain02/home/J2AD006/jxb06/jxc15-jxb06/image_conda.sif \
       /jmain02/home/J2AD006/jxb06/jxc15-jxb06/Block_ExperienceReplay/tasks/cm/jxc15/ber_Qbert_4.sh

sbatch --mem=100G \
       --gres=gpu:1 \
       --cpus-per-task=15 \
       --time=4-10:00:00 \
       -p small \
       /jmain02/home/J2AD006/jxb06/jxc15-jxb06/singbatch \
       /jmain02/home/J2AD006/jxb06/jxc15-jxb06/image_conda.sif \
       /jmain02/home/J2AD006/jxb06/jxc15-jxb06/Block_ExperienceReplay/tasks/cm/jxc15/pber_SpaceInvaders_4.sh
