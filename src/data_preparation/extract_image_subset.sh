#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=img_sub
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

embedding_count=$1

# if embedding count is empty default to 20
if [ -z "$embedding_count" ]
then
    echo "embedding count is empty, defaulting to 20"
    embedding_count=20
fi

echo "embedding count: $embedding_count"

python3 src/data_preparation/extract_image_subset.py -ec $embedding_count