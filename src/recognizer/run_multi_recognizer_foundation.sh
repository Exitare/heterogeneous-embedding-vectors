#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=m_r_f
#SBATCH --time=9-00:00:00
#SBATCH --partition=exacloud
#SBATCH --qos=long_jobs
#SBATCH --ntasks=1
#SBATCH --mem=384000
#SBATCH --cpus-per-task=4
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kirchgae@ohsu.edu

run_iteration=$1
cancer_types=$2


python3 src/recognizer/multi_cancer_recognizer_foundation.py -ri "${run_iteration}" -c ${cancer_types}