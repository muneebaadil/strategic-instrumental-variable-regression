# harris et. al settings. (clipping yes, selection no, random conversion matrix)
python college_admissions_experiments.py --n-cores 10 --num-repeat 10 --num-applicants 10000 --admit-all --applicants-per-round 1 --clip --experiment-root experiments --experiment-name clipped-yes-selection-no-random-effort-matrix-d1 --stream --generate 1

# harris et. al + selection (clipping yes, selection yes, random conversion matrix) 
# need round parameters. 
ROUNDS=(100 500 1000)
for round in "${ROUNDS[@]}"; do
    python college_admissions_experiments.py --n-cores 10 --num-repeat 10 --num-applicants 10000 --applicants-per-round ${round} --clip --experiment-root experiments --experiment-name "clipped-yes-selection-yes-random-effort-matrix-d1-round${round}" --stream --generate 1
done