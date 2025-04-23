#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=c_h5df
#SBATCH --time=9-00:00:00
#SBATCH --partition=batch
#SBATCH --qos=long_jobs
#SBATCH --ntasks=1
#SBATCH --mem=512000
#SBATCH --cpus-per-task=2
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=kirchgae@ohsu.edu

image_embedding_count=$1

# if image embedding count is empty default to -1
if [ -z "$image_embedding_count" ]
then
    echo "image embedding count is empty, defaulting to -1"
    image_embedding_count=-1
fi

echo "image embedding count: $image_embedding_count"

python3 src/data_preparation/0_14_create_h5.py -c BRCA LUAD STAD BLCA COAD THCA -ec $image_embedding_count