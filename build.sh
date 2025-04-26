#!/usr/bin/env bash
# build.sh
set -o errexit

# Create necessary directories
mkdir -p static
mkdir -p templates

# Make sure the templates directory contains the index.html file
if [ ! -f "templates/index.html" ]; then
    echo "Error: templates/index.html not found!"
    exit 1
fi

# Install Python dependencies
pip install -r requirements.txt