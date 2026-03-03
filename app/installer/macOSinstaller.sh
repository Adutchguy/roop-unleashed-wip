#!/bin/bash

# This script checks and installs all dependencies which are needed to run roop-unleashed. After that, it clones the repo.
# Execute this easily with /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/PJF16/roop-unleashed/master/installer/macOSinstaller.sh)

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo "Starting check and installation process of dependencies for roop-unleashed"

# Check if Homebrew is installed
if ! command_exists brew; then
    echo "Homebrew is not installed. Starting installation..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "Homebrew is already installed."
fi

# Update Homebrew
echo "Updating Homebrew..."
brew update

# Check if Python 3.11 is installed
if brew list --versions python@3.11 >/dev/null; then
    echo "Python 3.11 is already installed."
else
    echo "Python 3.11 is not installed. Installing it now..."
    brew install python@3.11
fi

# Check if python-tk@3.11 is installed
if brew list --versions python-tk@3.11 >/dev/null; then
    echo "python-tk@3.11 is already installed."
else
    echo "python-tk@3.11 is not installed. Installing it now..."
    brew install python-tk@3.11
fi

# Check if ffmpeg is installed
if command_exists ffmpeg; then
    echo "ffmpeg is already installed."
else
    echo "ffmpeg is not installed. Installing it now..."
    brew install ffmpeg
fi

# Check if git is installed
if command_exists git; then
    echo "git is already installed."
else
    echo "git is not installed. Installing it now..."
    brew install git
fi

# Clone the repository
REPO_URL="https://github.com/C0untFloyd/roop-unleashed.git"
REPO_NAME="roop-unleashed"

echo "Cloning the repository $REPO_URL..."
git clone $REPO_URL

# Check if the repository was cloned successfully
if [ -d "$REPO_NAME" ]; then
    echo "Repository cloned successfully. Changing into directory $REPO_NAME..."
    cd "$REPO_NAME"
else
    echo "Failed to clone the repository."
fi

echo "Check and installation process completed."

