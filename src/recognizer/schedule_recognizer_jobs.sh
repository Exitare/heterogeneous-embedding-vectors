

for i in $(seq 2 9)
do
  # run it 30 times
  for j in $(seq 1 30)
  do
    sbatch ./src/recognizer/run_recognizer.sh $i $j
  done
done