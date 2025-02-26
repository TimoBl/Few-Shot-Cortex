#!/bin/bash
#SBATCH --job-name="DL+DiReCT"
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6G

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate DeepScan

MY_PID=$$
echo "MY PID: ${MY_PID}"

papermill pre-processing.ipynb --kernel DeepScan --log-output >> pre.log