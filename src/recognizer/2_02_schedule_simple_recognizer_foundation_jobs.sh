  # run it 30 times
for i in $(seq 1 30)
do
  sbatch ./src/recognizer/2_02_run_simple_recognizer_foundation.sh $i
done
