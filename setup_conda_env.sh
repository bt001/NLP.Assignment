#!/bin/bash

# Display banner
echo "📦 NLP Project Environment Setup"
echo "-------------------------------"

# Ask for target directory
read -p "Enter the full path where you want to set up the environment (leave blank for current directory): " ENV_DIR

# Use current directory if none provided
if [ -z "$ENV_DIR" ]; then
  ENV_DIR=$(pwd)
  echo "📂 No directory specified. Using current directory: $ENV_DIR"
else
  echo "📂 Using specified directory: $ENV_DIR"
fi

# Copy environment.yml into the target directory
cp environment.yml "$ENV_DIR"

# Move into that directory
cd "$ENV_DIR" || {
  echo "❌ Failed to enter directory: $ENV_DIR"
  exit 1
}

# Create the Conda environment
echo "⚙️  Creating Conda environment from environment.yml..."
conda env create -f environment.yml

# Confirm creation
if [ $? -eq 0 ]; then
  echo "✅ Environment created successfully!"
  echo "💡 To activate it, run:"
  echo "   conda activate nlp-env"
else
  echo "❌ Failed to create the environment. Please check environment.yml and conda installation."
fi
