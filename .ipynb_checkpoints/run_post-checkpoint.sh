#!/bin/bash
#SBATCH --job-name="DL+DiReCT"
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=12G

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate DeepScan

MY_PID=$$
echo "MY PID: ${MY_PID}"

papermill post-processing.ipynb --kernel DeepScan --log-output >> out_deer/direct.log
