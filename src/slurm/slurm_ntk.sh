#!/bin/bash
#SBATCH --job-name=large_ntk_feature_learn
#SBATCH --output /home/ddz5/Desktop/sds659/src/slurm/logs/large_1000_particles_ntk_feature_learn_%j.log
#SBATCH --mail-type=ALL                            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=david.zhang.ddz5@yale.edu                   # Where to send mail
#SBATCH --partition gpu
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                        # Run on a single CPU
#SBATCH --gres=gpu:1                               # start with 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256gb                                 # Job memory request
#SBATCH --time=1-12:00:00                          # Time limit hrs:min:sec
date;hostname

/gpfs/radev/home/ddz5/.conda/envs/452_env/bin/python /home/ddz5/Desktop/sds659/src/ntk_feature_learning.py
