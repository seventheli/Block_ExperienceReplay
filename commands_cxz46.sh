sbatch --mem=100G \
       --gres=gpu:1 \
       --cpus-per-task=15 \
       --time=4-10:00:00 \
       -p small \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/Block_ExperienceReplay/tasks/Test_apex_2.sh

sbatch --mem=100G \
       --gres=gpu:1 \
       --cpus-per-task=15 \
       --time=4-10:00:00 \
       -p small \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/singbatch \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/image_conda.sif \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/Block_ExperienceReplay/tasks/cm/pber_Qbert_8.sh

sbatch --mem=100G \
       --gres=gpu:1 \
       --cpus-per-task=15 \
       --time=4-10:00:00 \
       -p small \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/singbatch \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/image_conda.sif \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/Block_ExperienceReplay/tasks/cm/ber_BeamRider_8.sh

sbatch --mem=100G \
       --gres=gpu:1 \
       --cpus-per-task=15 \
       --time=4-10:00:00 \
       -p small \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/singbatch \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/image_conda.sif \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/Block_ExperienceReplay/tasks/cm/per_Qbert.sh

sbatch --mem=100G \
       --gres=gpu:1 \
       --cpus-per-task=15 \
       --time=4-10:00:00 \
       -p small \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/singbatch \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/image_conda.sif \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/Block_ExperienceReplay/tasks/cm/er_SpaceInvaders.sh

sbatch --mem=100G \
       --gres=gpu:1 \
       --cpus-per-task=15 \
       --time=4-10:00:00 \
       -p small \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/singbatch \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/image_conda.sif \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/Block_ExperienceReplay/tasks/cm/er_Breakout.sh

sbatch --mem=100G \
       --gres=gpu:1 \
       --cpus-per-task=15 \
       --time=4-10:00:00 \
       -p small \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/singbatch \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/image_conda.sif \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/Block_ExperienceReplay/tasks/cm/pber_BeamRider_8.sh

sbatch --mem=100G \
       --gres=gpu:1 \
       --cpus-per-task=15 \
       --time=4-10:00:00 \
       -p small \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/singbatch \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/image_conda.sif \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/Block_ExperienceReplay/tasks/cm/ber_SpaceInvaders_8.sh

sbatch --mem=100G \
       --gres=gpu:1 \
       --cpus-per-task=15 \
       --time=4-10:00:00 \
       -p small \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/singbatch \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/image_conda.sif \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/Block_ExperienceReplay/tasks/cm/pber_Breakout_8.sh

sbatch --mem=100G \
       --gres=gpu:1 \
       --cpus-per-task=15 \
       --time=4-10:00:00 \
       -p small \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/singbatch \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/image_conda.sif \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/Block_ExperienceReplay/tasks/cm/er_Qbert.sh

sbatch --mem=100G \
       --gres=gpu:1 \
       --cpus-per-task=15 \
       --time=4-10:00:00 \
       -p small \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/singbatch \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/image_conda.sif \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/Block_ExperienceReplay/tasks/cm/ber_Qbert_8.sh

sbatch --mem=100G \
       --gres=gpu:1 \
       --cpus-per-task=15 \
       --time=4-10:00:00 \
       -p small \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/singbatch \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/image_conda.sif \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/Block_ExperienceReplay/tasks/cm/er_BeamRider.sh

sbatch --mem=100G \
       --gres=gpu:1 \
       --cpus-per-task=15 \
       --time=4-10:00:00 \
       -p small \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/singbatch \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/image_conda.sif \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/Block_ExperienceReplay/tasks/cm/ber_Breakout_8.sh

sbatch --mem=100G \
       --gres=gpu:1 \
       --cpus-per-task=15 \
       --time=4-10:00:00 \
       -p small \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/singbatch \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/image_conda.sif \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/Block_ExperienceReplay/tasks/cm/per_BeamRider.sh

sbatch --mem=100G \
       --gres=gpu:1 \
       --cpus-per-task=15 \
       --time=4-10:00:00 \
       -p small \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/singbatch \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/image_conda.sif \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/Block_ExperienceReplay/tasks/cm/per_SpaceInvaders.sh

sbatch --mem=100G \
       --gres=gpu:1 \
       --cpus-per-task=15 \
       --time=4-10:00:00 \
       -p small \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/singbatch \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/image_conda.sif \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/Block_ExperienceReplay/tasks/cm/pber_SpaceInvaders_8.sh
