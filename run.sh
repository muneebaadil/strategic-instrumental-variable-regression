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

# our settings, but no clipping, and increased bias.
ROUNDS=(1000 500 100)
for round in "${ROUNDS[@]}"; do
  python college_admissions_experiments.py --n-cores 50 --num-repeat 50 --num-applicants 100000 --applicants-per-round ${round} --experiment-root experiments --experiment-name "clipped-no-selection-yes-fixed-effort-matrix-d1-scaled-duplicates-round${round}-obias10" --stream --generate 1 --fixed-effort-conversion --scaled-duplicates sequence --o-bias 10 --b1bias 800 --b2bias 1.8
done
