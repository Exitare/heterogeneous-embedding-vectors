
cancer_types=$1
amount_of_summed_embeddings=$2
# upper bound is the maximum walk distance
upper_bound=$3



# if upper bound not set, set to 10
if [ -z "$upper_bound" ]
then
  echo "Upper bound not set, setting to 10"
  upper_bound=10
fi

# if summed embeddings count not set, set to 100000
if [ -z "$amount_of_summed_embeddings" ]
then
  echo "Amount of summed embeddings count not set, setting to 100000."
  amount_of_summed_embeddings=100000
fi

# run it 30 times
for iteration in $(seq 1 30)
do
    ./src/recognizer/models/2_03_run_multi_recognizer.sh -1 $iteration "${cancer_types}" $amount_of_summed_embeddings 0.0
done