"""
Configuration file for CodeBuddy project
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
TRAINING_DIR = PROJECT_ROOT / "training"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
UTILS_DIR = PROJECT_ROOT / "utils"

# Create directories if they don't exist
for dir_path in [MODEL_DIR, DATA_DIR, TRAINING_DIR, EVALUATION_DIR, UTILS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    "code_t5_model": "Salesforce/codet5-base",  # CodeT5 is RoBERTa-based, not T5
    "code_t5_max_length": 512,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "max_code_length": 1024,
    "max_explanation_length": 256,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 50,
    "num_beams": 4,
    "early_stopping": True,
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 8,
    "learning_rate": 5e-5,
    "num_epochs": 10,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 4,
    "max_grad_norm": 1.0,
    "save_steps": 1000,
    "eval_steps": 500,
    "logging_steps": 100,
}

# Vector database configuration
VECTOR_DB_CONFIG = {
    "index_type": "faiss",
    "dimension": 384,  # all-MiniLM-L6-v2 embedding dimension
    "metric": "cosine",
    "nlist": 100,
    "nprobe": 10,
}

# Dataset configuration
DATASET_CONFIG = {
    "code_search_net": {
        "name": "code-search-net/code_search_net",
        "subset": "python",
        "split": "train",
        "max_samples": 100000,
    },
    "docstring": {
        "name": "microsoft/DocString",
        "split": "train",
        "max_samples": 50000,
    },
    "human_eval": {
        "name": "openai_humaneval",
        "split": "test",
        "max_samples": 164,
    },
    "mbpp": {
        "name": "Muennighoff/mbpp",
        "split": "train",
        "max_samples": 10000,
    },
}

# Alias for backward compatibility
DATA_CONFIG = DATASET_CONFIG

# Evaluation configuration
EVALUATION_CONFIG = {
    "metrics": ["bleu", "bertscore", "exact_match"],
    "reference_column": "explanation",
    "prediction_column": "generated_explanation",
    "code_column": "code",
    "question_column": "question",
    "answer_column": "answer",
}

# Streamlit configuration
STREAMLIT_CONFIG = {
    "page_title": "CodeBuddy - Code Explanation Generator",
    "page_icon": "🤖",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# API configuration (for future use)
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": False,
    "reload": True,
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": PROJECT_ROOT / "logs" / "codebuddy.log",
}

# Create logs directory
(PROJECT_ROOT / "logs").mkdir(exist_ok=True)
