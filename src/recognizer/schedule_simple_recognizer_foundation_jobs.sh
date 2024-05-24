  # run it 30 times
for j in $(seq 1 30)
do
  sbatch ./src/recognizer/run_simple_recognizer_foundation.sh $i $j
done
