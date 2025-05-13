#!/bin/bash -l
#SBATCH --job-name=bo_imide
#SBATCH --partition=gpu8          # 按集群实际分区修改
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=90G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-7%8           # 32 个任务，最多 8 张 A100 并发
#SBATCH --output=logs/bo_%A_%a.out
#SBATCH --error=logs/bo_%A_%a.err

source activate
conda activate tfvaenew

# 为本 task 生成独立随机种子与输出目录
SEED=$((1000 + SLURM_ARRAY_TASK_ID))
OUTDIR=../result/optimization/bo_batch/run_${SLURM_ARRAY_TASK_ID}
export TMPDIR=/scratch/${USER}/${SLURM_JOB_ID}/${SLURM_ARRAY_TASK_ID}
mkdir -p "$TMPDIR" "$OUTDIR"

python bayes_uc.py \
  --seed  ${SEED} \
  --outdir ${OUTDIR}                   # 允许 sbatch 后续附加额外 CLI
