# ./src/recognizer/schedule_multi_recognizer_jobs.sh "brca laml"
cancer_types=$1

for i in $(seq 2 9)
do
  # run it 30 times
  for j in $(seq 1 30)
  do
    sbatch ./src/recognizer/2_03_run_multi_recognizer.sh $i $j "${cancer_types}"
  done
done