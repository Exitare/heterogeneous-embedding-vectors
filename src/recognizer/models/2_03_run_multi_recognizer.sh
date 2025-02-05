#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=m_r
#SBATCH --time=9-00:00:00
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --mem=128000
#SBATCH --cpus-per-task=4
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kirchgae@ohsu.edu

walk_distance=$1
run_iteration=$2
cancer_types=$3
amount_of_summed_embeddings=$4
noise_ratio=$5

# if no noise ratio is provided, set to 0.0
if [ -z "$noise_ratio" ]
then
  echo "Noise ratio not set, setting to 0.0"
  noise_ratio=0.0
fi

echo "Walk distance: $walk_distance"
echo "Run iteration: $run_iteration"
echo "Cancer types: $cancer_types"
echo "Amount of summed embeddings: $amount_of_summed_embeddings"
echo "Noise ratio: $noise_ratio"

#echo src/recognizer/models/2_03_multi_cancer_recognizer.py -w "${walk_distance}" -ri "${run_iteration}" -c ${cancer_types}
python3 src/recognizer/models/2_03_multi_cancer_recognizer_nc.py -w "${walk_distance}" -ri "${run_iteration}" -c ${cancer_types} -a "${amount_of_summed_embeddings}" -n "${noise_ratio}"