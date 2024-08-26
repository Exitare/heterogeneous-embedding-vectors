#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=m_r
#SBATCH --time=9-00:00:00
#SBATCH --partition=exacloud
#SBATCH --qos=long_jobs
#SBATCH --ntasks=1
#SBATCH --mem=128000
#SBATCH --cpus-per-task=32
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kirchgae@ohsu.edu

walk_distance=$1
run_iteration=$2
cancer_types=$3

python3 src/recognizer/2_03_multi_cancer_recognizer.py -w "${walk_distance}" -ri "${run_iteration}" -c ${cancer_types}