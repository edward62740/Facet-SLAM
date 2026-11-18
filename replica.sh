# Facet-SLAM on replica
# Usage: bash replica.sh [more main.py flags, e.g. --seed 1]
OUTPUT_PATH="output/Replica"
PYTHON=${PYTHON:-.venv/bin/python}
cd "$(dirname "$0")"

run_()
{
    local dataset=$1
    shift
    echo "run $dataset"
    $PYTHON main.py --dataset $dataset\
                    --output_dir $OUTPUT_PATH\
                    --eval "$@"
}

for dataset in room0 room1 room2 office0 office1 office2 office3 office4; do
    run_ $dataset "$@"
done
