# motivation experiments
ROUNDS=(1 10 100 500 1000)
for round in "${ROUNDS[@]}"; do
    # for round 1, this is the origianl tsls settings.
    python college_admissions_experiments.py --num-applicants 10000 --num-repeat 10 \
    --admit-all --applicants-per-round $round --experiment-name 'tsls-round$round-admitall' \
    --experiment-root experiments/motivation

    python college_admissions_experiments.py --num-applicants 10000 --num-repeat 10 \
    --applicants-per-round $round --experiment-name 'tsls-round$round-admitsome' \
    --experiment-root experiments/motivation
done
