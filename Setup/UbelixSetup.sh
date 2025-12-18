module load CUDA/12.1.1
export TCNN_CUDA_ARCHITECTURES=90 
export CUDA_HOME=/software.9/software/CUDA/11.8.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
module spider anaconda
module load Anaconda3/2024.02-1
which conda
eval "$(/software.9/software/Anaconda3/2024.02-1/bin/conda shell.bash hook)"
