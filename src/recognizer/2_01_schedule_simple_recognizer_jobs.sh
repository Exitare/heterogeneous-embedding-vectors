

for i in $(seq 2 9)
do
  # run it 30 times
  for j in $(seq 1 30)
  do
    sbatch ./src/recognizer/2_01_run_simple_recognizer.sh $i $j
  done
done