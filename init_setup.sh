#!/bin/bash

# Exit if any command fails
set -e

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

# echo "Installing dependencies..."
# pip install -e ".[dev]"

echo "makding directory structure..."
python template.py

echo "Setup complete ✅"
