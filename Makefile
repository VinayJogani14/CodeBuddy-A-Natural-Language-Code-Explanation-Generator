# Makefile for CodeBuddy Project

.PHONY: help install setup test demo run clean download-data train evaluate

# Default target
help:
	@echo "CodeBuddy: Natural Language to Code Explanation Generator"
	@echo "======================================================"
	@echo ""
	@echo "Available commands:"
	@echo "  install     - Install project dependencies"
	@echo "  setup       - Complete project setup (install + configure)"
	@echo "  test        - Run project tests"
	@echo "  demo        - Run project demo"
	@echo "  run         - Start Streamlit app"
	@echo "  download-data - Download datasets"
	@echo "  train       - Train the model"
	@echo "  evaluate    - Evaluate the model"
	@echo "  clean       - Clean project files"
	@echo "  help        - Show this help message"

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

# Complete project setup
setup:
	@echo "Setting up CodeBuddy project..."
	python scripts/setup.py

# Run tests
test:
	@echo "Running project tests..."
	python scripts/test_project.py

# Run demo
demo:
	@echo "Running CodeBuddy demo..."
	python scripts/demo.py

# Start Streamlit app
run:
	@echo "Starting CodeBuddy Streamlit app..."
	streamlit run app.py

# Download datasets
download-data:
	@echo "Downloading datasets..."
	python scripts/download_datasets.py

# Create sample data for testing
sample-data:
	@echo "Creating sample data..."
	python scripts/download_datasets.py --create_sample

# Train the model
train:
	@echo "Training CodeT5 model..."
	python training/train_code_explainer.py

# Evaluate the model
evaluate:
	@echo "Evaluating model..."
	python evaluation/evaluator.py

# Clean project files
clean:
	@echo "Cleaning project files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf logs/*
	rm -rf evaluation_results/*
	rm -rf training_outputs/*
	@echo "Cleanup complete!"

# Create virtual environment
venv:
	@echo "Creating virtual environment..."
	python -m venv venv
	@echo "Virtual environment created. Activate it with:"
	@echo "  source venv/bin/activate  # On Unix/macOS"
	@echo "  venv\\Scripts\\activate     # On Windows"

# Install in development mode
dev-install:
	@echo "Installing in development mode..."
	pip install -e .

# Format code
format:
	@echo "Formatting code..."
	black .
	isort .

# Lint code
lint:
	@echo "Linting code..."
	pylint models/ data/ training/ evaluation/ utils/ scripts/ app.py config.py

# Check code quality
check: format lint test

# Quick start (setup + demo)
quick-start: setup demo

# Development workflow
dev: install test demo

# Production setup
prod: install test download-data
