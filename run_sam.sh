#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <image_path> [output_mask_path]"
    exit 1
fi

IMAGE_PATH="$1"
OUTPUT_MASK="${2:-$(dirname $IMAGE_PATH)/mask_$(basename $IMAGE_PATH)}"

# Save current conda environment
CURRENT_ENV="$CONDA_DEFAULT_ENV"

# Create output directory
mkdir -p "$(dirname "$OUTPUT_MASK")"

# Activate SAM environment
echo "Switching to SAM environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sam

# Run SAM inference
echo "Running SAM inference on $IMAGE_PATH..."
cd /home/zy3722/work/FoundationPose
python sam_infer.py --image_path "$IMAGE_PATH" --output_path "$OUTPUT_MASK"

# Switch back to original environment
echo "Switching back to $CURRENT_ENV..."
conda deactivate

# Output the mask path
echo "MASK_PATH:$OUTPUT_MASK"