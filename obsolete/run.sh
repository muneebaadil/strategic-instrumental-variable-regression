# # harris et. al original settings. (clipping yes, selection no, fixed-effort-conversion and scaled duplicates no.)
# python college_admissions_experiments.py --n-cores 10 --num-repeat 10 --num-applicants 10000 --admit-all --applicants-per-round 1 --clip --experiment-root experiments --test-run --stream --generate 1

# # harris et. al with. selection (clipping yes, selection yes, fixed-effort-conversion and scaled duplicates no.) 
# # need round parameters. 
# ROUNDS=(100)
# for round in "${ROUNDS[@]}"; do
  # python college_admissions_experiments.py --n-cores 10 --num-repeat 10 --num-applicants 10000 --applicants-per-round ${round} --clip --experiment-root experiments-new-theory --experiment-name "harris-with-selection-round${round}" --stream --generate 1 # --fixed-effort-conversion --scaled-duplicates sequence
# done

# # harris et. al with selection + common effort conversion matrix.
# ROUNDS=(100)
# for round in "${ROUNDS[@]}"; do
  # python college_admissions_experiments.py --n-cores 10 --num-repeat 10 --num-applicants 10000 --applicants-per-round ${round} --clip --experiment-root experiments-new-theory --experiment-name "harris-with-selection-common-conversion-round${round}" --stream --generate 1 --fixed-effort-conversion # --scaled-duplicates sequence
# done 

# # harris et. al with selection + common effort conversion + no clipping + scaled duplicates
# ROUNDS=(100)
# BIASES=(1.25 1.5 1.75 2)
# for bias in "${BIASES[@]}"; do
  # python college_admissions_experiments.py --n-cores 10 --num-repeat 10 --num-applicants 10000 --applicants-per-round 100 --experiment-root experiments-new-theory --experiment-name "our-settings-bias${bias}" --stream --generate 1 --fixed-effort-conversion --scaled-duplicates sequence --b-bias $bias
# done

# # multi-env settings
# ROUNDS=(100)
# BIASES=(1.25 1.5 1.75 2)
# for bias in "${BIASES[@]}"; do
  # python college_admissions_experiments.py --n-cores 10 --num-repeat 10 --num-applicants 10000 --applicants-per-round 100 --experiment-root experiments-new-theory --experiment-name "our-settings-bias${bias}-multi" --stream --generate 1 --fixed-effort-conversion --scaled-duplicates sequence --b-bias $bias --num-envs 2 --pref geometric --prob 0.5
# done

# # validate number of envs.
# NUM_ENVS=(1 2 4 8 16)
# for num_envs in "${NUM_ENVS[@]}"; do
  # python college_admissions_experiments.py --n-cores 10 --num-repeat 10 --num-applicants 10000 --applicants-per-round 100 --experiment-root experiments-new-theory --experiment-name "our-settings-multi-numenvs${num_envs}" --generate 1 --fixed-effort-conversion --scaled-duplicates sequence --b-bias 2 --num-envs $num_envs 
# done

# validate the probability of geometric distribution
# PROBS=(0.1 0.3 0.5 0.7 0.9)
# for prob in "${PROBS[@]}"; do
  # python college_admissions_experiments.py --n-cores 10 --num-repeat 10 --num-applicants 10000 --applicants-per-round 100 --experiment-root experiments-new-theory --experiment-name "our-settings-multi-prob${prob}" --generate 1 --fixed-effort-conversion --scaled-duplicates sequence --b-bias 2 --num-envs 4  --pref geometric --prob $prob 
# done

python college_admissions_experiments.py --n-cores 1 --num-repeat 1 --num-applicants 10000 --applicants-per-round 100 --fixed-effort-conversion --scaled-duplicates sequence --b-bias 2 --num-envs 2 --pref uniform --experiment-root protocol-exps --experiment-name protocol --generate 1 --stream --save-dataset
python college_admissions_experiments.py --n-cores 1 --num-repeat 1 --num-applicants 100000 --applicants-per-round 1000 --fixed-effort-conversion --scaled-duplicates sequence --b-bias 2 --num-envs 2 --pref uniform --experiment-root protocol-exps --test-run --generate 1 --stream
python college_admissions_experiments.py --n-cores 1 --num-repeat 1 --num-applicants 10000 --applicants-per-round 100 --fixed-effort-conversion --scaled-duplicates sequence --b-bias 2 --num-envs 2 --pref uniform --experiment-root protocol-exps --experiment-name no-protocol --generate 1 --stream --no-protocol --save-dataset