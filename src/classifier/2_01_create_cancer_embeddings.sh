#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=cc_emb
#SBATCH --time=9-00:00:00
#SBATCH --partition=exacloud
#SBATCH --qos=long_jobs
#SBATCH --ntasks=1
#SBATCH --mem=64000
#SBATCH --cpus-per-task=32
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kirchgae@ohsu.edu

# sbatch ./src/classifier/2_01_create_cancer_embeddings.sh "BRCA BLCA LUAD STAD THCA COAD"

cancer_types=$1

python3 ./src/classifier/2_01_create_cancer_embeddings.py -c ${cancer_types}