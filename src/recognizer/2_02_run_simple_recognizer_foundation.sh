#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=s_r_f
#SBATCH --time=9-00:00:00
#SBATCH --partition=exacloud
#SBATCH --qos=long_jobs
#SBATCH --ntasks=1
#SBATCH --mem=640000
#SBATCH --cpus-per-task=32
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kirchgae@ohsu.edu

run_iteration=$1
upper_walk_distance=$2
summed_embeddings_count=$3


#echo src/recognizer/2_02_simple_recognizer_foundation.py -ri "${run_iteration}" -uwd "${upper_walk_distance}"
python3 src/recognizer/2_02_simple_recognizer_foundation.py -ri "${run_iteration}" -uwd "${upper_walk_distance}" -sec "${summed_embeddings_count}"