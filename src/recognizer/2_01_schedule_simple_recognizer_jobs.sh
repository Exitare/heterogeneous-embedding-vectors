
# add range of values to iterate through
upper_bound=$1


# if upper bound not set, set to 10
if [ -z "$upper_bound" ]
then
  echo "Upper bound not set, setting to 10"
  upper_bound=10
fi


for i in $(seq 2 $upper_bound)
do
  # run it 30 times
  for j in $(seq 1 30)
  do
    sbatch ./src/recognizer/2_01_run_simple_recognizer.sh $i $j
  done
done