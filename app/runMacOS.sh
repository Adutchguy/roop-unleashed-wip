#!/bin/bash

# Check if we are in the correct repository directory
if [ ! -f "run.py" ]; then
    echo "run.py not found!"
    exit 1
fi

# Create a hidden Python 3.11 virtual environment in the .venv folder
VENV_DIR=".venv"

# Check if Python 3.11 is installed
if ! brew list --versions python@3.11 >/dev/null; then
    echo "Python 3.11 is not installed. Please install it first."
    exit 1
fi

# Use Python 3.11 to create the virtual environment
echo "Creating a virtual environment using Python 3.11..."
python3.11 -m venv $VENV_DIR

# Activate the virtual environment
echo "Activating the virtual environment..."
source "$VENV_DIR/bin/activate"

# Check if the activation was successful
if [ "$VIRTUAL_ENV" != "" ]; then
    echo "Virtual environment activated successfully."
else
    echo "Failed to activate the virtual environment."
    exit 1
fi

# Install dependencies from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Skipping dependency installation."
fi

# Run roop-unleashed. This can take a while - especially at first startup...
echo "Running the run.py script..."
python run.py

# Deactivate the virtual environment after execution
echo "Deactivating the virtual environment..."
deactivate