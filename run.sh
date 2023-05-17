# # harris et. al original settings. (clipping yes, selection no, fixed-effort-conversion and scaled duplicates no.)
# python college_admissions_experiments.py --n-cores 10 --num-repeat 10 --num-applicants 10000 --admit-all --applicants-per-round 1 --clip --experiment-root experiments --test-run --stream --generate 1

# # harris et. al with. selection (clipping yes, selection yes, fixed-effort-conversion and scaled duplicates no.) 
# # need round parameters. 
ROUNDS=(100)
for round in "${ROUNDS[@]}"; do
  python college_admissions_experiments.py --n-cores 10 --num-repeat 10 --num-applicants 10000 --applicants-per-round ${round} --clip --experiment-root experiments-new-theory --experiment-name "harris-with-selection-round${round}" --stream --generate 1 # --fixed-effort-conversion --scaled-duplicates sequence
done

# harris et. al with selection + common effort conversion matrix.
ROUNDS=(100)
for round in "${ROUNDS[@]}"; do
  python college_admissions_experiments.py --n-cores 10 --num-repeat 10 --num-applicants 10000 --applicants-per-round ${round} --clip --experiment-root experiments-new-theory --experiment-name "harris-with-selection-common-conversion-round${round}" --stream --generate 1 --fixed-effort-conversion # --scaled-duplicates sequence
done 

# harris et. al with selection + common effort conversion + no clipping + scaled duplicates
ROUNDS=(100)
BIASES=(1.25 1.5 1.75 2)
for bias in "${BIASES[@]}"; do
  python college_admissions_experiments.py --n-cores 10 --num-repeat 10 --num-applicants 10000 --applicants-per-round 100 --experiment-root experiments-new-theory --experiment-name "our-settings-bias${bias}" --stream --generate 1 --fixed-effort-conversion --scaled-duplicates sequence --b-bias $bias
done