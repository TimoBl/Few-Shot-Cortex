#!/bin/bash
#SBATCH --job-name="DL+DiReCT"
#SBATCH --time=3:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=14G
#SBATCH --mail-user=timo.blattner@students.unibe.ch
#SBATCH --mail-user=end,fail

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate DeepScan

MY_PID=$$
echo "MY PID: ${MY_PID}"

papermill baseline.ipynb --kernel DeepScan --log-output >> out_sotalia/run.log