sbatch --mem=160G \
       --gres=gpu:1 \
       --cpus-per-task=20 \
       --time=4-10:00:00 \
       -p small \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/Block_ExperienceReplay/tasks/dper_MiniGrid-LavaCrossingS9N3.sh

sleep 14400

sbatch --mem=160G \
       --gres=gpu:1 \
       --cpus-per-task=20 \
       --time=4-10:00:00 \
       -p small \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/Block_ExperienceReplay/tasks/dpber_MiniGrid-DistShift2.sh


sbatch --mem=160G \
       --gres=gpu:1 \
       --cpus-per-task=20 \
       --time=4-10:00:00 \
       -p small \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/Block_ExperienceReplay/tasks/dper_MiniGrid-DistShift2.sh

sleep 14400

sbatch --mem=160G \
       --gres=gpu:1 \
       --cpus-per-task=20 \
       --time=4-10:00:00 \
       -p small \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/Block_ExperienceReplay/tasks/dpber_MiniGrid-Empty-8x8.sh


sbatch --mem=160G \
       --gres=gpu:1 \
       --cpus-per-task=20 \
       --time=4-10:00:00 \
       -p small \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/Block_ExperienceReplay/tasks/dper_MiniGrid-Empty-8x8.sh

sleep 14400

sbatch --mem=160G \
       --gres=gpu:1 \
       --cpus-per-task=20 \
       --time=4-10:00:00 \
       -p small \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/Block_ExperienceReplay/tasks/dpber_MiniGrid-ObstructedMaze-1Dlhb.sh


sbatch --mem=160G \
       --gres=gpu:1 \
       --cpus-per-task=20 \
       --time=4-10:00:00 \
       -p small \
       /jmain02/home/J2AD006/jxb06/cxz46-jxb06/Block_ExperienceReplay/tasks/dper_MiniGrid-ObstructedMaze-1Dlhb.sh

