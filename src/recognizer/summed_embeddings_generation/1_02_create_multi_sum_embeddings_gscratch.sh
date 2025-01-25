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
#SBATCH --mail-type=FAIL,BEGIN,END
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
cd $SCRATCH_PATH || exit 1

# Construct the file name based on selected cancers
file_name=$(echo "${selected_cancers}" | tr ' ' '_').h5
embeddings_file_path="/home/groups/EllrottLab/heterogeneous-embedding-vectors/results/embeddings/${file_name}"
script_file_path="/home/groups/EllrottLab/heterogeneous-embedding-vectors/src/recognizer/summed_embeddings_generation/1_02_create_multi_cancer_sum_embeddings.py"

# Save folder construction
results_folder="/home/groups/EllrottLab/heterogeneous-embedding-vectors/results/recognizer/summed_embeddings/multi/$(echo "${selected_cancers}" | tr ' ' '_')/${amount_of_summed_embeddings}/${noise_ratio}"
results_file="${walk_distance}_embeddings.h5"
gscratch_results_folder="${SCRATCH_PATH}/results/recognizer/summed_embeddings/multi/$(echo "${selected_cancers}" | tr ' ' '_')/${amount_of_summed_embeddings}/${noise_ratio}"

echo "File name: ${file_name}"
echo "SCRATCH_PATH: ${SCRATCH_PATH}"
echo "Embeddings file path: ${embeddings_file_path}"
echo "Original results folder: ${results_folder}"
echo "Results file: ${results_file}"
echo "gscratch_results_folder: ${gscratch_results_folder}"

# Check if embeddings file exists and copy to scratch
if [ -f "${embeddings_file_path}" ]; then
    echo "Copying ${embeddings_file_path} to scratch"
    cp "${embeddings_file_path}" "${SCRATCH_PATH}/"
else
    echo "Error: File ${embeddings_file_path} does not exist."
    exit 1
fi

# Check if Python script exists and copy to scratch
if [ -f "${script_file_path}" ]; then
    echo "Copying Python script to scratch"
    cp "${script_file_path}" "${SCRATCH_PATH}/"
else
    echo "Error: Python script ${script_file_path} does not exist."
    exit 1
fi

# Run the Python script with the specified arguments
echo "Running Python script"
echo "walk_distance: ${walk_distance}"
echo "selected_cancers: ${selected_cancers}"
echo "amount_of_summed_embeddings: ${amount_of_summed_embeddings}"
echo "noise_ratio: ${noise_ratio}"
echo "load_path: ${SCRATCH_PATH}/${file_name}"

srun python3 "${SCRATCH_PATH}/1_02_create_multi_cancer_sum_embeddings.py" \
    -a "${amount_of_summed_embeddings}" \
    -w "${walk_distance}" \
    -c "${selected_cancers}" \
    -n "${noise_ratio}" \
    --load_path "${SCRATCH_PATH}/${file_name}"

# Copy only the results file to the save folder
echo "Copying results file (${results_file}) back to ${save_folder}"
mkdir -p "${save_folder}" # Create the save folder if it doesn't exist

if [ -f "${gscratch_results_folder}/${results_file}" ]; then
    cp "${gscratch_results_folder}/${results_file}" "${results_folder}/"
    echo "Results file copied successfully."
else
    echo "Error: Results file ${results_file} not found in scratch."
    exit 1
fi

# Back out of the scratch directory before cleanup
cd ../ || exit 1

# Clean up scratch directory
echo ""
echo "************************************"
echo "Cleaning up workdir"
srun rmdir-scratch.sh
