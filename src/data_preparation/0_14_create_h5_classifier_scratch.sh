#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=h5_classifier
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
#SBATCH --gres=disk:2512 # Request scratch space in GB

selected_cancers=$1
dry_run=$2
combined_cancers=$(echo "${selected_cancers}" | tr ' ' '_')
output_file="results/embeddings/${combined_cancers}_classifier.h5"

echo $selected_cancers
echo $combined_cancers
echo $output_file

# Create working directory in scratch
echo ""
echo "************************************"
echo "Making working directory in scratch"
srun /usr/local/bin/mkdir-scratch.sh
SCRATCH_PATH="/mnt/scratch/${SLURM_JOB_ID}"
cd $SCRATCH_PATH || exit 1


echo "Scratch Path: $SCRATCH_PATH"

echo "Creating folders"
mkdir -p $SCRATCH_PATH/results/embeddings/rna/${combined_cancers}
mkdir -p $SCRATCH_PATH/results/embeddings/images
mkdir -p $SCRATCH_PATH/results/embeddings/annotations/${combined_cancers}

ls -l

echo "Copying files to path $SCRATCH_PATH"

echo "Copying rna files..."
# copy embeddings/rna, embeddings/images, annotations
cp -r /home/groups/EllrottLab/heterogeneous-embedding-vectors/results/embeddings/rna/${combined_cancers} $SCRATCH_PATH/results/embeddings/rna/
echo "Copying image files..."
cp -r /home/groups/EllrottLab/heterogeneous-embedding-vectors/results/embeddings/images/ $SCRATCH_PATH/results/embeddings/images/
echo "Copying annotation files..."
cp -r /home/groups/EllrottLab/heterogeneous-embedding-vectors/results/embeddings/annotations/${combined_cancers}  $SCRATCH_PATH/results/embeddings/annotations
echo "Copying script file..."
cp -r /home/groups/EllrottLab/heterogeneous-embedding-vectors/src/data_preparation/0_14_create_h5_classifier.py $SCRATCH_PATH/
echo "Copying mutation embeddings..."
cp /home/groups/EllrottLab/heterogeneous-embedding-vectors/results/mutation_embeddings.csv $SCRATCH_PATH/results/embeddings

ls -l $SCRATCH_PATH/results/embeddings
ls -l $SCRATCH_PATH/results/embeddings/images/
ls -l $SCRATCH_PATH/results/embeddings/rna/
ls -l $SCRATCH_PATH/results/embeddings/annotations/


echo "Running script from path: $SCRATCH_PATH/0_14_create_h5_classifier.py"

srun python3 $SCRATCH_PATH/0_14_create_h5_classifier.py -c BRCA LUAD STAD BLCA COAD THCA

echo "Copying files back"
cp $SCRATCH_PATH/$output_file /home/groups/EllrottLab/heterogeneous-embedding-vectors/results/embeddings/

# Back out of the scratch directory before cleanup
cd ../ || exit 1

# Clean up scratch directory
echo ""
echo "************************************"
echo "Cleaning up workdir"
srun rmdir-scratch.sh
