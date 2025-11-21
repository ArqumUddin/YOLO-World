#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Install from https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment
echo "Creating conda environment..."
if conda env list | grep -q "^yolo-world "; then
    echo "Environment 'yolo-world' exists. Remove and recreate? (y/n)"
    read -r reply
    [[ $reply =~ ^[Yy]$ ]] && conda env remove -n yolo-world -y && conda env create -f environment.yml
else
    conda env create -f environment.yml
fi

# Setup third_party/mmyolo
echo "Setting up third_party/mmyolo..."
mkdir -p third_party
if [ -d "third_party/mmyolo" ]; then
    echo "mmyolo exists. Update? (y/n)"
    read -r reply
    if [[ $reply =~ ^[Yy]$ ]]; then
        rm -rf third_party/mmyolo
        git clone --depth 1 --filter=blob:none --sparse https://github.com/open-mmlab/mmyolo.git third_party/mmyolo
        cd third_party/mmyolo
        git sparse-checkout set configs
        cd "$SCRIPT_DIR"
    fi
else
    git clone --depth 1 --filter=blob:none --sparse https://github.com/open-mmlab/mmyolo.git third_party/mmyolo
    cd third_party/mmyolo
    git sparse-checkout set configs
    cd "$SCRIPT_DIR"
fi

# Create weights directory and download models
echo "Creating weights directory..."
mkdir -p weights

echo "Downloading weights..."
declare -A WEIGHTS=(
    ["yolo_world_v2_s_stage1.pth"]="https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/s_stage1-4466ab94.pth"
    ["yolo_world_v2_s_stage2.pth"]="https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/s_stage2-4466ab94.pth"
    ["yolo_world_v2_m_stage1.pth"]="https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/m_stage1-7e1e5299.pth"
    ["yolo_world_v2_m_stage2.pth"]="https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/m_stage2-9987dcb1.pth"
    ["yolo_world_v2_l_stage1.pth"]="https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/l_stage1-7d280586.pth"
    ["yolo_world_v2_l_stage2.pth"]="https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/l_stage2-b3e3dc3f.pth"
    ["yolo_world_v2_x_stage1.pth"]="https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/x_stage1-62b674ad.pth"
)

for weight_file in "${!WEIGHTS[@]}"; do
    [ -f "weights/$weight_file" ] && echo "✓ $weight_file exists" && continue
    echo "Downloading $weight_file..."
    curl -L -o "weights/$weight_file" "${WEIGHTS[$weight_file]}"
done

echo ""
echo "Setup complete! Activate with: conda activate yolo-world"
