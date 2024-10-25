#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=s_e_g
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

walk_distance=$1 # the walk distance, aka how many embeddings are summed
summed_embeddings_count=$2 # how many summed vectors should be created
noise=$3 # the noise level

python3 src/recognizer/1_01_create_simple_sum_embeddings.py -i "${summed_embeddings_count}" -w "${walk_distance}" -n "${noise}"