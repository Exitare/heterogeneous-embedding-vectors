#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=b_r
#SBATCH --time=23:00:00
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --mem=128000
#SBATCH --cpus-per-task=4
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kirchgae@ohsu.edu

cancer_types=$1
walk_distance=$2
run_iteration=$3
amount_of_summed_embeddings=$4
noise_ratio=$5
multi=$6

# if multi set to true
if [ "$multi" = "true" ]; then
    multi="--multi"
else
    multi=""
fi

python3 src/recognizer/models/1_01_baseline_model.py  -c ${cancer_types}  -w "${walk_distance}" -ri "${run_iteration}" -a "${amount_of_summed_embeddings}" -n "${noise_ratio}" ${multi}