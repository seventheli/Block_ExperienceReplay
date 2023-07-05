#!/bin/bash

# set the number of nodes
SBATCH --nodes=1

# set max wallclock time
SBATCH --time=01:00:00

# set name of job
SBATCH --job-name=test_rl_benchmark

# set number of GPUs
SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
SBATCH --mail-type=ALL

# send mail to this address
SBATCH --mail-user=b5048266@newcastle.ac.uk

# run the application
myCode
