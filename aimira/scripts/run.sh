#!/bin/bash

## SBATCH --partition=LKEBgpu
## SBATCH --gres=gpu:RTX6000:1
## SBATCH --ntasks=1
## SBATCH --cpus-per-task=4
## SBATCH --mem=128GB

#!/bin/bash
#SBATCH --partition=amd-gpu-long,gpu-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
#SBATCH -t 06:00:00
#SBATCH --mem-per-gpu=120G
#SBATCH -e results/logs/slurm-%j.err
#SBATCH -o results/logs/slurm-%j.out

# module purge
# module add library/cuda/11.2/gcc.8.3.1

hostname
echo "Cuda devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi
sleep 10

export PATH=/home/jjia/.conda/envs/py38/bin:$PATH
eval "$(conda shell.bash hook)"

conda activate py38
job_id=$SLURM_JOB_ID
slurm_dir=results/logs
echo job_id is $job_id
##cp script.sh ${slurm_dir}/slurm-${job_id}.shs
# git will not detect the current file because this file may be changed when this job was run



scontrol write batch_script ${job_id} ${slurm_dir}/slurm-${job_id}_args.sh
idx=0; export CUDA_VISIBLE_DEVICES=$idx; python -u run.py 2>${slurm_dir}/slurm-${job_id}_${idx}_err.txt 1>${slurm_dir}/slurm-${job_id}_${idx}_out.txt --outfile=${slurm_dir}/slurm-${job_id}_$idx --hostname="$(hostname)" --jobid=${job_id} --total_folds=5 --remark="total_folds=5, nested_valid"
