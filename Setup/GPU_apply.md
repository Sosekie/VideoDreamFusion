gpu-usage
squeue -u $USER
scancel <JOBID>
ssh <JOBID>


rsync -avz ./ cf23h027@submit02.unibe.ch:~/threestudio_H100/

rsync -avz \
  --exclude='.git' \
  --exclude='__pycache__' \
  cf23h027@submit02.unibe.ch:~/threestudio_H100/ ./

srun --partition=gpu --qos=job_interactive --gpus=h100:1 --cpus-per-task=4 --mem=90G --time=08:00:00 --pty bash

module load CUDA/12.1.1
export TCNN_CUDA_ARCHITECTURES=90 
export CUDA_HOME=/software.9/software/CUDA/11.8.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
module spider anaconda
module load Anaconda3/2024.02-1
which conda
eval "$(/software.9/software/Anaconda3/2024.02-1/bin/conda shell.bash hook)"