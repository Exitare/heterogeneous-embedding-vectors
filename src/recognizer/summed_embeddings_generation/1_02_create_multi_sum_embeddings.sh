#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=m_e_g
#SBATCH --time=23:00:00
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --mem=128000
#SBATCH --cpus-per-task=2
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kirchgae@ohsu.edu

walk_distance=$1
selected_cancers=$2
amount_of_summed_embeddings=$3
noise_ratio=$4

python3 src/recognizer/summed_embeddings_generation/1_02_create_multi_cancer_sum_embeddings.py  -a "${amount_of_summed_embeddings}" -w "${walk_distance}" -c ${selected_cancers} -n "${noise_ratio}"