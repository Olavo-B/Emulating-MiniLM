#!/bin/bash

# Initial configurations
PROJECT_NAME=$(basename "$(realpath .)")
IMAGE_NAME=$(echo "$PROJECT_NAME" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
FULL_PATH=$(realpath .)
DOCKERFILE_PATH="./Dockerfile"

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

# Function to display help
show_help() {
    echo -e "\e[33mUsage:\e[0m ./install.sh [options]"
    echo -e "\nOptions:"
    echo -e "  -h, --help        Show this help message and exit."
    echo -e "  -n, --name NAME   Set the Docker image name (default: current folder name)."
    echo -e "\nExample:\n  ./install.sh --name my-image"
}

# Argument parsing
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -n|--name)
            IMAGE_NAME="$2"
            shift
            shift
            ;;
        *)
            echo_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Change permissions for run.sh
chmod +x run.sh

# Check if Dockerfile exists
if [ ! -f "$DOCKERFILE_PATH" ]; then
    echo_error "Dockerfile not found at $DOCKERFILE_PATH."
    exit 1
fi

# Build Docker image
echo_info "Building Docker image with name: $IMAGE_NAME"
docker build -t "$IMAGE_NAME" .

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo_success "Docker image '$IMAGE_NAME' built successfully."
else
    echo_error "Failed to build Docker image."
    exit 1
fi

# Generate json file with the image name and info
echo_info "Generating image info file: image_info.json"
echo -e $IMAGE_NAME > misc/image_info.txt


# Run the container with default settings
echo_info "Running container: $IMAGE_NAME"
docker run --gpus all --rm -it \
    --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
    --name "$IMAGE_NAME" \
    -v "$FULL_PATH:/app" \
    "$IMAGE_NAME" bash
