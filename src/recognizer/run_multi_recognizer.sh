#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=mlt_recognizer
#SBATCH --time=9-00:00:00
#SBATCH --partition=exacloud
#SBATCH --qos=long_jobs
#SBATCH --ntasks=1
#SBATCH --mem=128000
#SBATCH --cpus-per-task=4
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kirchgae@ohsu.edu

total_embeddings=$1
run_iteration=$2
cancer_types=$3


python3 src/recognizer/multi_cancer_recognizer.py -e "${total_embeddings}" -ri "${run_iteration}" -c ${cancer_types}