#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=c_m_e
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

selected_cancers=$1
amount_of_summed_embeddings=$2
noise_ratio=$3

python3 src/recognizer/summed_embeddings_generation/combine_summed_embeddings.py  -a "${amount_of_summed_embeddings}" -c ${selected_cancers} -n "${noise_ratio}"