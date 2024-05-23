#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=m_e_g
#SBATCH --time=9-00:00:00
#SBATCH --partition=exacloud
#SBATCH --qos=long_jobs
#SBATCH --ntasks=1
#SBATCH --mem=64000
#SBATCH --cpus-per-task=4
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kirchgae@ohsu.edu

total_embeddings=$1
cancer_types=$2

python3 src/embedding_aggregation/create_multi_cancer_sum_embeddings.py  -i 1000000 -e "${total_embeddings}" -c ${cancer_types}