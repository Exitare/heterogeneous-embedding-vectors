#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=r_w
#SBATCH --time=9-00:00:00
#SBATCH --partition=exacloud
#SBATCH --qos=long_jobs
#SBATCH --ntasks=1
#SBATCH --mem=512000
#SBATCH --cpus-per-task=32
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kirchgae@ohsu.edu

run_iteration=$1
random_walks=$2
cancer_types=$3

python3 src/nodevec/random_walks.py -i "${run_iteration}" -w "${random_walks}" -c ${cancer_types}