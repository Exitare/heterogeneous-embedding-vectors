  # run it 30 times
for i in $(seq 1 30)
do
  sbatch ./src/recognizer/run_multi_recognizer_foundation.sh $i
done
