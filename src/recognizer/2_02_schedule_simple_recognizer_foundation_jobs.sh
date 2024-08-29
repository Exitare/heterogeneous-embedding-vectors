upper_walk_distance=$1

# if upper bound not set, set to 10
if [ -z "$upper_walk_distance" ]
then
  echo "Upper walk distance not set, setting to 10"
  upper_walk_distance=10
fi

# run it 30 times
for i in $(seq 1 30)
do
  sbatch ./src/recognizer/2_02_run_simple_recognizer_foundation.sh $i $upper_walk_distance
done
