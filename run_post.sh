#!/bin/bash
#SBATCH --job-name="DL+DiReCT"
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12G

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate DeepScan

MY_PID=$$
echo "MY PID: ${MY_PID}"

papermill post-processing.ipynb -p NAME "Seal5" -p SRC_DIR "/storage/homefs/tb19m004/UniBe/MasterThesis/experiments/baseline/out_zero/out_Seal5/" --kernel DeepScan --log-output >> out_zero/out_Seal5/run.log


