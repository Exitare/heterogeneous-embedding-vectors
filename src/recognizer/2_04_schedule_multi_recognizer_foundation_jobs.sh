# ./src/recognizer/schedule_multi_recognizer_foundation_jobs.sh "brca laml"

cancer_types=$1

# run it 30 times
for i in $(seq 1 30)
do
  sbatch ./src/recognizer/2_04_run_multi_recognizer_foundation.sh $i "${cancer_types}"
done
