#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=c_e_g
#SBATCH --time=9-00:00:00
#SBATCH --partition=batch
#SBATCH --qos=long_jobs
#SBATCH --ntasks=1
#SBATCH --mem=256000
#SBATCH --cpus-per-task=2
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=kirchgae@ohsu.edu

selected_cancers=$1
walk_distance=$2
amount_of_walks=$3

python ./src/tmb_classifier/summed_embeddings_generation/3_01_create_summed_embeddings.py -c ${selected_cancers} -w $walk_distance -a $amount_of_walks
