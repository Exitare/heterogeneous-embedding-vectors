#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=h5_classifier
#SBATCH --time=9-00:00:00
#SBATCH --partition=batch
#SBATCH --qos=long_jobs
#SBATCH --ntasks=1
#SBATCH --mem=64000
#SBATCH --cpus-per-task=2
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=kirchgae@ohsu.edu

python3 src/data_preparation/0_14_create_h5_classifier.py -c BRCA LUAD STAD BLCA COAD THCA