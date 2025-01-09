#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=classifier
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

#./src/classifier/4_01_cancer_classifier.sh "BRCA BLCA LUAD STAD THCA COAD"


cancers=$1
walk_distance=$2
walk_amount=$3
iteration=$4

# Check if cancers is empty
if [ -z "$cancers" ]; then
  echo "Error: No cancers provided. Please specify the cancers to classify."
  exit 1
fi

# if walk_distance is empty, set to 1
if [ -z "$walk_distance" ]; then
  walk_distance=3
  echo "walk_distance is empty, setting to 3"
fi

# if walk_amount is empty, set to 1
if [ -z "$walk_amount" ]; then
  walk_amount=3
  echo "walk_amount is empty, setting to 3"
fi

# call script
python src/classifier/4_01_cancer_classifier.py -c ${cancers} -i ${iteration} -w ${walk_distance} -a ${walk_amount}
