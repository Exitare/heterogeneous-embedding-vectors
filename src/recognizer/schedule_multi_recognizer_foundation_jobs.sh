# ./src/recognizer/schedule_multi_recognizer_foundation_jobs.sh "blca brca"

cancer_types=$1

# run it 30 times
for i in $(seq 1 30)
do
  sbatch ./src/recognizer/run_multi_recognizer_foundation.sh $i "${cancer_types}"
done
