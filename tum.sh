# Facet-SLAM on TUM rgbd
# Usage: bash tum.sh [more main.py flags, e.g. --seed 1]
OUTPUT_PATH="output/TUM"
PYTHON=${PYTHON:-.venv/bin/python}
cd "$(dirname "$0")"

run_()
{
    local dataset=$1
    shift
    echo "run $dataset"
    $PYTHON main.py --dataset $dataset\
                    --output_dir $OUTPUT_PATH\
                    --max_fps 30\
                    --eval "$@"
}

for dataset in fr1_desk fr2_xyz fr3_office; do
    run_ $dataset "$@"
done
