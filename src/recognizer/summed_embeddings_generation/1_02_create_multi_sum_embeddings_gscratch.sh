#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=m_e_g
#SBATCH --time=9-00:00:00
#SBATCH --partition=batch
#SBATCH --qos=long_jobs
#SBATCH --ntasks=1
#SBATCH --mem=128000
#SBATCH --cpus-per-task=2
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kirchgae@ohsu.edu
#SBATCH --gres=disk:2048 # Request scratch space in GB

# Input arguments
walk_distance=$1
selected_cancers=$2
amount_of_summed_embeddings=$3
noise_ratio=$4

# Create working directory in scratch
echo ""
echo "************************************"
echo "Making working directory in scratch"
srun /usr/local/bin/mkdir-scratch.sh
SCRATCH_PATH="/mnt/scratch/${SLURM_JOB_ID}"
cd $SCRATCH_PATH

# Construct the file name based on selected cancers
file_name=$(echo "${selected_cancers}" | tr ' ' '_' ).h5
source_file_path="./results/embeddings/${file_name}"

echo "File name: ${file_name}"
echo "Source file path: ${source_file_path}"

# Copy only the specific file to the scratch directory
if [ -f "${source_file_path}" ]; then
    echo "Copying ${source_file_path} to scratch"
    cp "${source_file_path}" "${SCRATCH_PATH}/"
else
    echo "Error: File ${source_file_path} does not exist."
    exit 1
fi

# Run your Python script with the scratch path as the load path
echo "Running Python script"
srun python3 src/recognizer/summed_embeddings_generation/1_02_create_multi_cancer_sum_embeddings.py \
    -a "${amount_of_summed_embeddings}" \
    -w "${walk_distance}" \
    -c "${selected_cancers}" \
    -n "${noise_ratio}" \
    --load_path "${SCRATCH_PATH}/${file_name}" # Pass the specific file in scratch as the load path

# Back out of the scratch directory before cleanup
cd ../

# Clean up scratch directory
echo ""
echo "************************************"
echo "Cleaning up workdir"
srun /usr/local/bin/rmdir-scratch.sh
