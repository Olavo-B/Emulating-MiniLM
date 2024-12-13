#!/bin/bash

# Initial configurations
PROJECT_NAME=$(basename "$(realpath .)")
IMAGE_NAME=$(echo "$PROJECT_NAME" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
FULL_PATH=$(realpath .)

# Function to display colored messages
echo_error() {
    echo -e "\e[31m[ERROR]\e[0m $1"
}
echo_success() {
    echo -e "\e[32m[SUCCESS]\e[0m $1"
}
echo_info() {
    echo -e "\e[34m[INFO]\e[0m $1"
}

# Run the container with default settings
echo_info "Running container: $IMAGE_NAME"
docker run --gpus all --rm -it \
    --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
    --name "$IMAGE_NAME" \
    -v "$FULL_PATH:/app" \
    "$IMAGE_NAME" bash