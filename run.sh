# # harris et. al original settings. (clipping yes, selection no, fixed-effort-conversion and scaled duplicates no.)
# python college_admissions_experiments.py --n-cores 10 --num-repeat 10 --num-applicants 10000 --admit-all --applicants-per-round 1 --clip --experiment-root experiments --experiment-name clipped-yes-selection-no-random-effort-matrix-d1 --stream --generate 1

# # harris et. al with. selection (clipping yes, selection yes, fixed-effort-conversion and scaled duplicates no.) 
# # need round parameters. 
# ROUNDS=(100 500 1000)
# for round in "${ROUNDS[@]}"; do
  # python college_admissions_experiments.py --n-cores 10 --num-repeat 10 --num-applicants 10000 --applicants-per-round ${round} --clip --experiment-root experiments --experiment-name "clipped-yes-selection-yes-random-effort-matrix-d1-round${round}" --stream --generate 1
# done

# our settings. (clipping yes, selection yes, fixed-effort-conversion and scaled duplicates yes)
ROUNDS=(100 500 1000)
for round in "${ROUNDS[@]}"; do
  python college_admissions_experiments.py --n-cores 10 --num-repeat 10 --num-applicants 10000 --applicants-per-round ${round} --clip --experiment-root experiments --experiment-name "clipped-yes-selection-yes-fixed-effort-matrix-d1-scaled-duplicates-round${round}" --stream --generate 1 --fixed-effort-conversion --scaled-duplicates sequence
done