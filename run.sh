# harris et. al settings. (clipping yes, selection no, random conversion matrix)
python college_admissions_experiments.py --n-cores 1 --num-repeat 10 --num-applicants 10000 --admit-all --applicants-per-round 1 --clip --experiment-root experiments --experiment-name clipped-yes-selection-no-random-effort-matrix-d1-algo --stream

# harris et. al + selection (clipping yes, selection yes, random conversion matrix) 
# need round parameters. 
