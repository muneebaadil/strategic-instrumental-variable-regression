# # harris et. al original settings. (clipping yes, selection no, fixed-effort-conversion and scaled duplicates no.)
# python college_admissions_experiments.py --n-cores 100 --num-repeat 100 --num-applicants 10000 --admit-all --applicants-per-round 1 --clip --experiment-root experiments --experiment-name clipped-yes-selection-no-random-effort-matrix-d1 --stream --generate 1

# harris et. al with. selection (clipping yes, selection yes, fixed-effort-conversion and scaled duplicates no.) 
# need round parameters. 
# ROUNDS=(100 500 1000)
# for round in "${ROUNDS[@]}"; do
 # python college_admissions_experiments.py --n-cores 100 --num-repeat 100 --num-applicants 10000 --applicants-per-round ${round} --clip --experiment-root experiments --experiment-name "clipped-yes-selection-yes-random-effort-matrix-d1-round${round}" --stream --generate 1
# done

# our settings. (clipping yes, selection yes, fixed-effort-conversion and scaled duplicates yes)
# ROUNDS=(1000 500 100)
# for round in "${ROUNDS[@]}"; do
  # python college_admissions_experiments.py --n-cores 100 --num-repeat 100 --num-applicants 10000 --applicants-per-round ${round} --clip --experiment-root experiments --experiment-name "clipped-yes-selection-yes-fixed-effort-matrix-d1-scaled-duplicates-round${round}" --stream --generate 1 --fixed-effort-conversion --scaled-duplicates sequence
# done

# # our settings, but no clipping, and increased bias.
# ROUNDS=(1000)
# for round in "${ROUNDS[@]}"; do
  # python college_admissions_experiments.py --n-cores 50 --num-repeat 50 --num-applicants 100000 --applicants-per-round ${round} --experiment-root experiments  --stream --generate 1 --fixed-effort-conversion --scaled-duplicates sequence --o-bias 10 --b1bias 800 --b2bias 1.8 --test-run #--experiment-name "clipped-no-selection-yes-fixed-effort-matrix-d1-scaled-duplicates-round${round}-obias10"
# done

# validating baseline biases.
# b1biases=(800 600 400 200)
# b2biases=(1.8 1.3 0.9 0.4)
# for i in 0 1 2 3; do
  # b1bias=${b1biases[i]}
  # b2bias=${b2biases[i]}
  # # echo $b1bias $b2bias
  # python college_admissions_experiments.py --n-cores 50 --num-repeat 50 --num-applicants 100000 --applicants-per-round 1000 --experiment-root experiments-ablation --generate 1  --fixed-effort-conversion --scaled-duplicates sequence --o-bias 10 --b1bias $b1bias --b2bias $b2bias --experiment-name "b1bias${b1bias}-b2bias${b2bias}" --stream
# done


# obiases=(1 2 4 8 10)
# for i in 0 1 2 3 4; do
  # # b1bias=${b1biases[i]}
  # # b2bias=${b2biases[i]}
  # obias=${obiases[i]}
  # python college_admissions_experiments.py --n-cores 50 --num-repeat 50 --num-applicants 100000 --applicants-per-round 1000 --experiment-root experiments-ablation --generate 1  --fixed-effort-conversion --scaled-duplicates sequence --o-bias $obias --experiment-name "obias${obias}" --stream
# done



# obias=1
# python college_admissions_experiments.py --n-cores 50 --num-repeat 50 --num-applicants 100000 --applicants-per-round 1000 --experiment-root experiments-postproc --generate 1  --fixed-effort-conversion --scaled-duplicates sequence --o-bias 1 --experiment-name "clip" --stream --post-proc clip
# obias=1
# python college_admissions_experiments.py --n-cores 50 --num-repeat 50 --num-applicants 100000 --applicants-per-round 1000 --experiment-root experiments-postproc --generate 1  --fixed-effort-conversion --scaled-duplicates sequence --o-bias 1 --experiment-name "scale" --stream --post-proc scale